export type PineconeEnv = {
  apiKey: string;
  environment: string;
}

/*
  * This function is used to get the Pinecone environment variables from the
  * environment. It is used by the PineconeVectorStore constructor to initialize
  * the PineconeClient.
  * 
  * Looks for:
  * - PINECONE_API_KEY: API Key for the Pinecone API. Formatted as a UUID.
  * - PINECONE_API_ENVIRONMENT: The environment the index is running in. Can be
  *   found on the index's page. Ex: "us-west1-gcp"
  * 
  * @throws {Error} if the environment variables are not found.
*/
export function getPineconeConfigFromEnv(): PineconeEnv {
  try {
    return {
      apiKey: process.env["PINECONE_API_KEY"]!,
      environment: process.env["PINECONE_API_ENVIRONMENT"]!
    }
  } catch {
    throw new Error("Missing Pinecone API environment variables. To fix, either:\
    - PINECONE_API_KEY and PINECONE_API_ENVIRONMENT or .\
    - Pass an initialized PineconeClient to the constructor via the `pineconeClient` parameter. \
    ");
  }
}