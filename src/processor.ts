import * as fs from 'fs/promises';
import * as path from 'path';
import readline from 'readline';
import chalk from 'chalk';

import { createClient } from '@supabase/supabase-js';

import { TextLoader } from 'langchain/document_loaders/fs/text';
import { CharacterTextSplitter } from 'langchain/text_splitter';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { SupabaseVectorStore } from 'langchain/vectorstores/supabase';
import {
  ChatPromptTemplate,
  SystemMessagePromptTemplate,
} from 'langchain/prompts';
import { ChatOpenAI } from 'langchain/chat_models/openai';
import { LLMChain } from 'langchain/chains';
import { TiktokenModel, encoding_for_model } from 'tiktoken';

type TokenizerResult = {
  docs: any[];
  tokens: number;
  price: number;
};

class CodeProcessor {
  private OPENAI_API_KEY = '';
  private OPENAI_MODEL: string;
  private SUPABASE_URL = '';
  private SUPABASE_KEY = '';

  private excludeDir: string[];
  private excludeFiles: string[];
  private excludeExtensions: string[];

  constructor(
    openAIApiKey: string,
    openAIModel: TiktokenModel | string,
    supabaseUrl: string,
    supabaseApiKey: string
  ) {
    this.excludeDir = ['.git', 'node_modules', 'public', 'assets'];
    this.excludeFiles = ['package-lock.json', '.DS_Store'];
    this.excludeExtensions = [
      '.jpg',
      '.jpeg',
      '.png',
      '.gif',
      '.bmp',
      '.tiff',
      '.ico',
      '.svg',
      '.webp',
      '.mp3',
      '.wav',
    ];
    this.OPENAI_API_KEY = openAIApiKey;
    this.OPENAI_MODEL = openAIModel || 'gpt-4';
    this.SUPABASE_URL = supabaseUrl;
    this.SUPABASE_KEY = supabaseApiKey;
  }

  async traverseDirectory(dir: string): Promise<string[]> {
    const files = await fs.readdir(dir);
    let filePaths: string[] = [];

    for (let file of files) {
      if (
        this.excludeDir.includes(file) ||
        this.excludeFiles.includes(file) ||
        this.excludeExtensions.includes(path.extname(file))
      ) {
        continue;
      }

      const filePath = path.join(dir, file);
      const stat = await fs.stat(filePath);

      if (stat.isDirectory()) {
        const dirPaths = await this.traverseDirectory(filePath);
        filePaths = filePaths.concat(dirPaths);
      } else {
        filePaths.push(filePath);
      }
    }

    return filePaths;
  }

  countToken(words: string): number {
    const encoding = encoding_for_model(this.OPENAI_MODEL as TiktokenModel);
    const tokens = encoding.encode(words).length;
    encoding.free();
    return tokens;
  }

  calculatePrice(tokens: number, model: string = 'gpt-4'): number {
    // Define the prices per token for each model
    const token = 1000;
    // Define the prices per token for each model
    const pricePerToken = {
      // GPT-4 8K
      'gpt-4': {
        input: 0.03 / token,
        output: 0.06 / token,
      },
      // GPT-4 32K
      'gpt-4-32k': {
        input: 0.06 / token,
        output: 0.12 / token,
      },
      // GPT-3.5 Turbo 4K
      'gpt-3.5-turbo': {
        input: 0.0015 / token,
        output: 0.002 / token,
      },
      // GPT-3.5 Turbo 16K
      'gpt-3.5-turbo-16k': {
        input: 0.003 / token,
        output: 0.004 / token,
      },
    };

    const totalPrice =
      tokens * pricePerToken[model as keyof typeof pricePerToken].input;

    return totalPrice;
  }

  async tokenizer(dir: string): Promise<TokenizerResult> {
    const filePaths = await this.traverseDirectory(dir);

    let documents: any[] = [];

    for (const filePath of filePaths) {
      const loader = new TextLoader(filePath);
      const docs = await loader.load();
      documents.push(...docs);
    }

    const textSplitter = new CharacterTextSplitter({
      chunkSize: 2000,
      chunkOverlap: 200,
    });
    const docsSplitted = await textSplitter.splitDocuments(documents);

    let tokens = 0;

    for (let doc of docsSplitted) {
      const source = doc.metadata['source'];
      const cleanedSource = source.split('/').slice(1).join('/');
      doc.pageContent = `FILE NAME: ${cleanedSource}\n###\n${doc.pageContent.replace(
        '\u0000',
        ''
      )}`;
      tokens += await this.countToken(doc.pageContent);
    }

    const price = await this.calculatePrice(tokens);

    console.log(`! ${chalk.yellow.bold('Usage Estimate')}`);
    console.log(`> ${chalk.bold('Token:')} ${chalk.red(tokens)}`);
    console.log(`> ${chalk.bold('Price:')} ${chalk.green('$' + price)}`);

    return {
      docs: docsSplitted,
      tokens,
      price,
    };
  }

  async embed(docs: any[]) {
    const supabaseClient = createClient(this.SUPABASE_URL, this.SUPABASE_KEY);
    const embeddings = new OpenAIEmbeddings({
      openAIApiKey: this.OPENAI_API_KEY,
    });

    console.log('>>> Embeddings');
    console.log(embeddings);

    const vectorStore = await SupabaseVectorStore.fromDocuments(
      docs,
      embeddings,
      {
        client: supabaseClient,
        tableName: 'documents',
        queryName: 'match_documents',
      }
    );

    console.log(vectorStore);
  }

  askQuestion(question: string): Promise<string> {
    return new Promise((resolve) => {
      const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
      });

      rl.question(question, (answer) => {
        rl.close();
        resolve(answer);
      });
    });
  }

  async chat() {
    const supabaseClient = createClient(this.SUPABASE_URL, this.SUPABASE_KEY, {
      auth: { persistSession: false },
    });

    const embeddings = new OpenAIEmbeddings({
      openAIApiKey: this.OPENAI_API_KEY,
    });

    const vectorStore = new SupabaseVectorStore(embeddings, {
      client: supabaseClient,
      tableName: 'documents',
      queryName: 'match_documents',
    });

    while (true) {
      const query = await this.askQuestion(
        chalk.blue('AI: What question do you have about your repo?\n')
      );

      if (query.toLowerCase().trim() === 'exit') {
        console.log(chalk.green('Goodbye!'));
        break;
      }

      const matchedDocs = await vectorStore.similaritySearch(query);
      let codeStr = '';

      for (const doc of matchedDocs) {
        codeStr += doc.pageContent + '\n\n';
      }

      const template = `
        You are Codebase AI. You are a superintelligent AI that answers questions about codebases.
    
        You are:
        - helpful & friendly
        - good at answering complex questions in simple language
        - an expert in all programming languages
        - able to infer the intent of the user's question
    
        The user will ask a question about their codebase, and you will answer it.
    
        When the user asks their question, you will answer it by searching the codebase for the answer.
    
        Here is the user's question and code file(s) you found to answer the question:
    
        Question:
        ${query}
    
        Code file(s):
        ${codeStr}
        
        [END OF CODE FILE(S)]
    
        Now answer the question using the code file(s) above.
      `;

      const systemMessagePrompt =
        SystemMessagePromptTemplate.fromTemplate(template);
      const chatPrompt = ChatPromptTemplate.fromPromptMessages([
        systemMessagePrompt,
      ]);

      const chat = new ChatOpenAI({
        temperature: 0.5,
        openAIApiKey: this.OPENAI_API_KEY,
      });

      const chain = new LLMChain({
        llm: chat,
        prompt: chatPrompt,
      });

      const res = await chain.call({ code: codeStr, query });

      console.log(res.text);

      console.log('\n\n');
    }
  }
}

export default CodeProcessor;
