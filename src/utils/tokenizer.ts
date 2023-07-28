/*
* Helper class for tokenizing text.
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
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  tokenizer: any;

  constructor(modelNameOrPath: string, options: TokenizerConfig) {
    // eslint-disable-next-line @typescript-eslint/ban-ts-comment
    // @ts-ignore
    transformersEnv.allowRemoteModels = options.allowRemoteModels || false;
    transformersEnv.localModelPath = options.localModelPath;

    this.modelNameOrPath = modelNameOrPath;
  }

  async init() {
    this.tokenizer = await AutoTokenizer.from_pretrained(this.modelNameOrPath);
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  async tokenize(text: string, tokenizeOptions: TokenizeOptions): Promise<any> {
    const options = { ...defaultTokenizeOptions, ...tokenizeOptions };
    const tokens = await this.tokenizer(text, { ...options, max_length: options.maxLength });
    return tokens;
  }
}