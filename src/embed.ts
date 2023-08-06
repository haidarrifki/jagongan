import * as fs from 'fs/promises';
import * as path from 'path';
import chalk from 'chalk';

import { createClient } from '@supabase/supabase-js';

import { TextLoader } from 'langchain/document_loaders/fs/text';
import { CharacterTextSplitter } from 'langchain/text_splitter';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { SupabaseVectorStore } from 'langchain/vectorstores/supabase';
import { TiktokenModel, encoding_for_model } from 'tiktoken';

import { OPENAI_API_KEY, SUPABASE_URL, SUPABASE_KEY } from '@/constants.ts';

type TokenizerResult = {
  docs: any[];
  tokens: number;
  price: number;
};

async function traverseDirectory(dir: string): Promise<string[]> {
  // configure these to fit your needs
  const excludeDir = ['.git', 'node_modules', 'public', 'assets'];
  const excludeFiles = ['package-lock.json', '.DS_Store'];
  const excludeExtensions = [
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

  const files = await fs.readdir(dir);
  let filePaths: string[] = [];

  for (let file of files) {
    if (
      excludeDir.includes(file) ||
      excludeFiles.includes(file) ||
      excludeExtensions.includes(path.extname(file))
    ) {
      continue;
    }

    const filePath = path.join(dir, file);
    const stat = await fs.stat(filePath);

    if (stat.isDirectory()) {
      const dirPaths = await traverseDirectory(filePath);
      filePaths = filePaths.concat(dirPaths);
    } else {
      filePaths.push(filePath);
    }
  }

  return filePaths;
}

// Function to count the token
function countToken(words: string, model: TiktokenModel): number {
  const encoding = encoding_for_model(model);
  const tokens = encoding.encode(words).length;
  encoding.free();
  return tokens;
}

// Function to calculate the price
function calculatePrice(tokens: number, model: string = 'gpt-4'): number {
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

async function tokenizer(dir: string, model: string): Promise<TokenizerResult> {
  const filePaths = await traverseDirectory(dir);

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
    tokens += await countToken(doc.pageContent, model as TiktokenModel);
  }

  const price = await calculatePrice(tokens, model);

  console.log(`! ${chalk.yellow.bold('Usage Estimate')}`);
  console.log(`> ${chalk.bold('Token:')} ${chalk.red(tokens)}`);
  console.log(`> ${chalk.bold('Price:')} ${chalk.green('$' + price)}`);

  return {
    docs: docsSplitted,
    tokens,
    price,
  };
}

async function embed(docs: any[]) {
  const supabaseClient = createClient(SUPABASE_URL, SUPABASE_KEY);
  const embeddings = new OpenAIEmbeddings({
    openAIApiKey: OPENAI_API_KEY,
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

export { tokenizer, embed };
