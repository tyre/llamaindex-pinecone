import { SparseValues } from ".";

class PineconeEmbedder {
  async init() {
    console.log('init')
  }

  embed(text: string) {

    const sparseValues: SparseValues = { indices: [], values: [] };
  }
}

