import readline from 'readline';
import chalk from 'chalk';
import { createClient } from '@supabase/supabase-js';

import { LLMChain } from 'langchain/chains';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import {
  ChatPromptTemplate,
  SystemMessagePromptTemplate,
} from 'langchain/prompts';
import { SupabaseVectorStore } from 'langchain/vectorstores/supabase';
import { ChatOpenAI } from 'langchain/chat_models/openai';

import { OPENAI_API_KEY, SUPABASE_URL, SUPABASE_KEY } from '@/constants.ts';

const supabaseClient = createClient(SUPABASE_URL, SUPABASE_KEY, {
  auth: { persistSession: false },
});

const embeddings = new OpenAIEmbeddings({ openAIApiKey: OPENAI_API_KEY });
const vectorStore = new SupabaseVectorStore(embeddings, {
  client: supabaseClient,
  tableName: 'documents',
  queryName: 'match_documents',
});

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

const askQuestion = (question: string): Promise<string> => {
  return new Promise((resolve) => {
    rl.question(question, (answer) => {
      resolve(answer);
    });
  });
};

const chat = async () => {
  while (true) {
    const query = await askQuestion(
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

    // console.log(chalk.yellow('\n' + codeStr + '\n'));

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
      // streaming: true,
      // callbacks: [new ConsoleCallbackHandler()],
      temperature: 0.5,
      openAIApiKey: OPENAI_API_KEY,
      // verbose: true,
    });

    const chain = new LLMChain({
      llm: chat,
      prompt: chatPrompt,
      // callbacks: [new ConsoleCallbackHandler()],
    });

    const res = await chain.call({ code: codeStr, query });

    console.log(res.text);

    console.log('\n\n');
  }

  rl.close();
};

export { chat };
