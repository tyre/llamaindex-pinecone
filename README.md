# LlamaIndex integration with Pinecone

This repository contains a LlamaIndex-compatible vector store backed by [Pinecone](https://pinecone.io) indices.

## Installation

~~npm install llamaindex-pinecone~~ NPM package outdated until deps update. Until then:

- Clone the typescript llamaindex package: `https://github.com/run-llama/LlamaIndexTS`
- Clone this package `git clone git@github.com:tyre/llamaindex-pinecone.git`
- In this repository, update `package.json` to include `"llamaindex":"file:path/to/LlamaIndexTS/packages/core"`
- In your project, update the `package.json` to include: `"llamaindex":"file:path/to/LlamaIndexTS/packages/core", "llamaindex-pinecone": "file:path/to/llamaindex-pinecone",`

At the end of the day, you need the edge version of the `llamaindex` package (from source) and update this package to point to it. Your package must point to the HEAD versions of each.

### API config

To automatically have a working client initialized, site these environment variables:

- `PINECONE_API_KEY`: Your API Key 
- `PINECONE_ENVIRONMENT`: The environment of the pinecone instance you're using


## Usage

The heart and soul of this package is `PineconeVectorStore`. Let's see how that works.

### `VectorStore` interface compliance

`PineconeVectorStore` adheres to the `VectorStore` interface from `llamaindex` almost entirely. Here is the `VectorStore` interface:

```typescript
export interface VectorStore {
  storesText: boolean;
  isEmbeddingQuery?: boolean;
  client(): any;
  add(embeddingResults: NodeWithEmbedding[]): Promise<string[]>;
  delete(refDocId: string, deleteKwargs?: any): Promise<void>;
  query(query: VectorStoreQuery, kwargs?: any): Promise<VectorStoreQueryResult>;
  persist(persistPath: string, fs?: GenericFileSystem): Promise<void>;
}
```

### Creating a store

```typescript
import { PineconeVectorStore } from "llamaindex-pinecone";

// Initialize with the name of an index in Pinecone 
const vectorStore = new PineconeVectorStore({indexName: "speeches"});
```

### Adding a node to the store

Now we want to add a node to the store. This requires us to pass a node and its embedding. The below example has already been tokenized:

```typescript
import { Document } from "llamaindex";

const tongueTwister = new Document({text: "Peter Piper picked a peck of pickled peppers. A peck of pickled peppers Peter Piper picked. If Peter Piper picked a peck of pickled peppers, Where's the peck of pickled peppers Peter Piper picked?", id_: "peter-piper"})
const embedding = [
  1, 5310, 7362, 546, 18691, 263, 1236, 384, 310, 5839, 839, 1236,
  22437, 29889, 319, 1236, 384, 310, 5839, 839, 1236, 22437, 5310, 7362,
  546, 18691, 29889, 960, 5310, 7362, 546, 18691, 263, 1236, 384, 310,
  5839, 839, 1236, 22437, 29892, 6804, 29915, 29879, 278, 1236, 384, 310,
  5839, 839, 1236, 22437, 5310, 7362, 546, 18691, 29973
];
await vectorStore.add({ node: tongueTwister, emedding })
// => ["peter-piper]
```

We see that it has returned an array of node ids that were successfully upserted to Pinecone.

#### Metadata

By default, PineconeVectorStore will upsert metadata in the form of:
```
{
  nodeId: // .nodeId for the passed in node,
  ...node.metadata
}
```

To customize this, pass a function of type `(node: BaseNode) => PineconeMetadata` as the `extractPineconeMetadata` option for `add`. (`PineconeMetadata` is of type: `Record<string, string | number | boolean | Array<string>>;`)

```typescript
const metadataBuilder = (node) =>{ return { nextNodeId: node.nodeId + 1 } };
await vectorStore.add({ node: tongueTwister, emedding }, { extractPineconeMetadata: metadataBuilder });
```

#### Passing duplicate vectors

Note that the passing duplicate nodes—those with identical node ids—and embeddings will only create one vector in Pinecone, but the response will count both. The returned array will return the nodeId twice.

#### Batching

The call to `add` takes a set of options as its second argument. Calls to the Pinecone API are batched, in groups of 100 by default. Passing a `batchSize` changes this value:

```typescript
await vectorStore.add(nodesWithEmbeddings, { batchSize:  500 })
```

Changing the batch size possibly affects how many requests are made to Pinecone's API. You may want to fiddle with this if you are hitting rate limits. Note that Pinecone recommends batch sizes less than 1000.

#### Sparse values

Upsertion supports sparse values, if `includeSparseValues: true` in passed in. By default, the sparse values will be built for you with a rather naive count-the-frequencies method.

With the same node and embedding above:

```typescript
await vectorStore.upsert(nodesWithEmbeddings, { includeSparseValues: true });
```
This will result in a sparse values dictionary included in vector upsert that looks like this:

```typescript
{
  indices: [
    1, 263, 278, 310, 319, 384, 546,
    839, 960, 1236, 5310, 5839, 6804, 7362,
    18691, 22437, 29879, 29889, 29892, 29915, 29973
  ],
  values: [
    1, 2, 1, 4, 1, 4, 4,
    4, 1, 8, 4, 4, 1, 4,
    4, 4, 1, 2, 1, 1, 1
  ]
}
```


### Querying

#### `query`

`PineconeVectorStore` implements `query` for the sake of complying with the `VectorStore`. Because a full implementation would require re-building all nodes from the returned results, which is not implemented, `query` presently throws an error every time.

Use `queryAll` instead.

#### `queryAll`

This method is mostly analagous with `VectorStore.query` except that it returns Pinecone's matches rather than nodes.

```typescript
import { VectorStoreQuery, VectorStoreQueryMode } from 'llamaindex';

class PineconeVectorStoreQuery implements VectorStoreQuery {
  queryEmbedding?: number[];
  similarityTopK: number;
  docIds?: string[];
  queryStr?: string;
  mode: VectorStoreQueryMode;
  alpha?: number;
  filters?: MetadataFilters;
  mmrThreshold?: number;

  constructor(args: any) {
    Object.assign(this, args);
  }
}

const query = new PineconeVectorStoreQuery({
  similarityTopK: 3,
  queryEmbedding: [1,2,3,4,5],
  mode: VectorStoreQueryMode.SPARSE
});

await vectorStore.queryAll(query);
/** =>
 * {
 *   matches: [
 *     { id: "vector-1", score: 0.123 },
 *     { id: "vector-2", score: 0.32 },
*      { id: "vector-3", score: 0.119 }
 *   ]
 * }
```
#### Options

The second argument to `queryAll` is a dictionary of options. Those may include:

- `namespace`: the namespace to query. If not provided, will query the default namespace.
- `includeMetadata`: whether or not to include metadata in the response. Defaults to true.
- `includeValues`: whether or not to include matching vector values in the response. Defaults to true.
- `vectorId`: the id of a vector already in Pinecone that will be used as the query. Exclusive with `query.queryEmbedding`.

#### Return value

As we've seen, the response is an object with a key `matches` and an array of scored vectors. Each object contains:

- `id`: id of the vector
- `score`: the similarity to the query vector
- `values`: the vector, if `includeValues` was `true`
- `sparseValues`: the sparse values of the vector
- `metadata`: the metadata of the vector, if `includeMetadata` was `true`

### Fetching vectors

Simple stuff.

Note: this fetches vectors, not vectors for a node.
For nodes with an embedding <= the dimension of the index, that's the same as the `node.nodeId`.

```typescript
vectorStore.client.fetch(["peter-piper"], "Namespace (Optional: defaults to default namespace)")
```

### Deleting vectors

#### By node id

Deletes all vectors associated with the given node ids.

🚨 NOTE 🚨
This does not work on Starter plans, which don't support filters on delete operations. Use `deleteVectors` instead.

```typescript
await client.delete("peter-piper");
```

#### For multiple nodes

```typescript
const nodeIds = ["wordNode", "peter-piper"];
await client.deleteAll(nodeIds);
```

#### By vector id

If you have the vectors' ids handy, you're welcome to delete those specifically:

```typescript
const vectorIds = [
  "wordNode-0",
  "wordNode-1"
  "wordNode-2",
  "wordNode-3"
];
await client.deleteVectors(vectorIds, "Namespace (Optional: defaults to default namespace)");
```

### Customization

`PineconeVectorStore` works well out of the box. You might want some customization, thought.

#### Customizing the client

By default, these Pinecone variables are pulled from the environment:

- `PINECONE_API_KEY`: Your API Key 
- `PINECONE_ENVIRONMENT`: The environment of the pinecone instance you're using

If you'd like to bring your own client, the `client` option will gladly accept yours.

```typescript
import { PineconeClient } from "@pinecone-database/pinecone";

const myPineconeClient = new PineconeClient();
await myPineconeClient.init({ apiKey: "something secure", environment: "something environmental" });

const vectorStore = new PineconeVectorStore({ indexName: "UFO-files", client: myPineconeClient })
```

#### Customizing Sparse value generation

The naive sparse value generation is, as its name implies, naive. Other methods, like BM25 or SPLADE, may be more effective. The vector store supports passing a class that knows how to generate sparse values.

```typescript
import { SparseValues, SparseValueBuilder } from "llamaindex-pinecone";

class FancySparseValueBuilder implements SparseValueBuilder {
  embeddings: Array<number>;
  constructor(embeddings: Array<number>) {
    this.embeddings = embeddings;
  }

  build(): SparseValues {
    // Do fancy math to generate indices and values
    // Return a dictionary with those (aka `SparseValues`):
    return { indicies, values }
  }
}

// Now we can use it!
const vectorStore = PineconeVectorStore("fancy-documents", { sparseValueBuilder: FancySparseValueBuilder });
```

When calling `add` or `upsert` with `includeSparseValues: true`, that builder will be used to generate sparse values being sent to Pinecone.

## Advanced uses

### Upsert

Upsert is the underlying method that backs `add`. Let's look back at the example from `add`'s documentation above to see what else `upsert` tells us.



```typescript
import { Document } from "llamaindex";

const tongueTwister = new Document({text: "Peter Piper picked a peck of pickled peppers. A peck of pickled peppers Peter Piper picked. If Peter Piper picked a peck of pickled peppers, Where's the peck of pickled peppers Peter Piper picked?", id_: "peter-piper"})
const embedding = [
  1, 5310, 7362, 546, 18691, 263, 1236, 384, 310, 5839, 839, 1236,
  22437, 29889, 319, 1236, 384, 310, 5839, 839, 1236, 22437, 5310, 7362,
  546, 18691, 29889, 960, 5310, 7362, 546, 18691, 263, 1236, 384, 310,
  5839, 839, 1236, 22437, 29892, 6804, 29915, 29879, 278, 1236, 384, 310,
  5839, 839, 1236, 22437, 5310, 7362, 546, 18691, 29973
];

const nodesWithEmbeddings = [{node: tongueTwister, embedding}];
await vectorStore.upsert(nodesWithEmbeddings);
/**
 * => {
 *   upsertedNodeCount: 1,
 *   upsertedNodeIds: ["peter-piper"],
 *   upsertedVectorCount: 1,
 *   upsertedVectorByNode: { "peter-piper": ["peter-piper"] }
 *   failedNodeCount: 0,
 *   failedNodeIds: [],
 *   errors: []
 *  }
*/
```

It's a lot! Let's break down what's in there:

- upsertedNodeCount: the total number of nodes upserted,
- upsertedNodeIds: the node ids that were successfully upserted
- upsertedVectorCount: the total number of vectors upserted
- upsertedVectorByNode: a mapping of the node ids to the vectors that were upserted for that node
- failedNodeCount: the number of nodes that failed to *fully* upsert
- failedNodeIds: the ids of the nodes that failed to *fully* upsert
- errors: an array of errors that occurred during upsert

For most cases, `upsertedNodeCount` and `upsertedVectorCount` will be the same. See  the "Automatic Embedding Splitting" section below for the option to split nodes across multiple vectors.

#### Automatic Embedding Splitting

By default, the embedding for a node is expected to be exactly as long as the dimension of the index. If not, `PineconeVectorStore` will throw an error.

If you have a large embedding that you would like automatically, though naively, chunked into multiple vectors the size of the index's dimension, set `splitEmbeddingsByDimension` to `true`:


Here's a contrived example with a pinecone index whose dimension is `1`:

```typescript
const indexInfo = await myPineconeClient.Index("letters").describeIndexStats();
console.log(indexInfo.dimension)
// => 1

const node = TextNode({text: "word", id_:"wordNode"})
vectorStore.upsert([{ node , embedding: [23, 15, 18, 4]}], { splitEmbeddingsByDimension: true })
/**
 * => {
 *   upsertedNodeCount: 1,
 *   upsertedNodeIds: ["wordNode"],
 *   upsertedVectorCount: 4,
 *   upsertedVectorByNode: {
 *     "wordNode": [
 *       "wordNode-0",
 *       "wordNode-1",
 *       "wordNode-2",
 *       "wordNode-3"
 *     ]
 *   },
 *   failedNodeCount: 0,
 *   failedNodeIds: [],
 *   errors: []
 *  }
*/
```

The API request to Pinecone would look something like this:

```JSON
{
  "vectors": [
    { "id": "wordNode-0", "values":[23], "metadata": { "nodeId": "wordNode" } },
    { "id": "wordNode-1", "values":[15], "metadata": { "nodeId": "wordNode" } },
    { "id": "wordNode-2", "values":[18], "metadata": { "nodeId": "wordNode" } },
    { "id": "wordNode-3", "values":[4], "metadata": { "nodeId": "wordNode" } }
  ]
}
```

Notice that the vector has been split to fit the dimension of the index. The ids in Pinecone have been adapted to be indexed in order, prefixed by the node's id, and the metadata is preserved.

The node's id is always included in the metadata, so deleting the document handles cleaning up all related vectors automatically. Query's can still filter by that node id in the metadata.

##### With batching

Note that for batched requests, the vectors for a given node can be spread across multiple requests. These requests can succeed or fail independently, so it is possible for a given node id to be in both the upserted and failed lists. Since these are upserts, it is safe to retry the failed nodes.