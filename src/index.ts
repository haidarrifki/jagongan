#!/usr/bin/env node
import * as path from 'path';

import { program } from 'commander';
import { select, confirm, input } from '@inquirer/prompts';

import CodeProcessor from '@/processor.ts';

program
  .helpOption('-h, --help', 'display help for command')
  .addHelpCommand('help [command]', 'display help for specific command')
  .option('-c, --chat', 'start chat immediately after processing')
  .option('-d, --dir <dir>', 'directory to process');

program.on('--help', () => {
  console.log('');
  console.log('Examples:');
  console.log('Processing repo:');
  console.log('  $ jagongan --dir /path/to/dir');
  console.log('Chat after processing:');
  console.log('  $ jagongan --chat');
});

program.parse(process.argv);

const options = program.opts();

const run = async () => {
  const inputOpenAIKey = await input({
    message: 'Enter your OpenAI API key:',
  });

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

  if (options.chat) {
    await processor.chat();
  } else {
    let dir = '';

    if (options.dir) {
      if (path.isAbsolute(options.dir)) {
        dir = options.dir;
      } else {
        dir = path.join(process.cwd(), options.dir);
      }
    }

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
  }
};

run();
