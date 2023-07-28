/*
* Helper class for tokenizing text in tests.
* Expects that the model is either already local or present
* in the HuggingFace cache.
*/

import { AutoTokenizer, env as transformersEnv } from "@xenova/transformers";

type TokenizerConfig = {
  localModelPath: string;
  allowRemoteModels: boolean;
}

type TokenizeOptions = {
  padding?: boolean;
  truncation?: boolean;
  maxLength: number;
}

const defaultTokenizeOptions: Partial<TokenizeOptions> = {
  padding: true,
  truncation: false,
}

export class Tokenizer {
  modelNameOrPath: string;
  tokenizer: any;

  constructor(modelNameOrPath: string, options: TokenizerConfig) {
    // @ts-ignore
    transformersEnv.allowRemoteModels = options.allowRemoteModels || false;
    transformersEnv.localModelPath = options.localModelPath;

    this.modelNameOrPath = modelNameOrPath;
  }

  async init() {
    this.tokenizer = await AutoTokenizer.from_pretrained(this.modelNameOrPath);
  }

  async tokenize(text: string, tokenizeOptions: TokenizeOptions): Promise<any> {
    const options = { ...defaultTokenizeOptions, ...tokenizeOptions };
    let tokens = await this.tokenizer(text, { ...options, max_length: options.maxLength });
    return tokens;
  }
}