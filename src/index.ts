#!/usr/bin/env node
import * as path from 'path';
import { select, confirm, input } from '@inquirer/prompts';

import CodeProcessor from '@/processor.ts';

const [, , ...args] = process.argv;

let dir = '';
if (path.isAbsolute(args[0])) {
  dir = args[0];
} else {
  dir = path.join(process.cwd(), args[0]);
}

const run = async () => {
  const inputOpenAIKey = await input({ message: 'Enter your OpenAI API key:' });
  const inputSupabaseUrl = await input({
    message: 'Enter your Supabase Project URL:',
  });
  const inputSupabaseKey = await input({
    message: 'Enter your Supabase Project API key:',
  });

  const model = await select({
    message: 'Select an OpenAI model',
    choices: [
      {
        name: 'gpt-4',
        value: 'gpt-4',
        description:
          'GPT-4: The latest model from OpenAI with improved performance and features.',
      },
      {
        name: 'gpt-4-32k',
        value: 'gpt-4-32k',
        description:
          'GPT-4 32K: The latest model with extended 32K token context for handling larger inputs.',
      },
      {
        name: 'gpt-3.5-turbo',
        value: 'gpt-3.5-turbo',
        description:
          'GPT-3.5 Turbo: A powerful and efficient model that offers a balance between performance and cost.',
      },
      {
        name: 'gpt-3.5-turbo-16k',
        value: 'gpt-3.5-turbo-16k',
        description:
          'GPT-3.5 Turbo 16K: An older model with extended 16K token context for handling larger inputs.',
      },
    ],
  });

  const processor = new CodeProcessor(
    inputOpenAIKey,
    model,
    inputSupabaseUrl,
    inputSupabaseKey
  );

  const { docs } = await processor.tokenizer(dir);

  const isConfirmed = await confirm({
    message: 'Are you sure you want continue?',
  });

  if (isConfirmed) {
    await processor.embed(docs);
    console.log('> Done.');
    await processor.chat();
  } else {
    console.log('Bye!');
  }
};

run();
