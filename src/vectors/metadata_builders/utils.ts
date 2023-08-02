export function validateMetadata(key: string, value: unknown): void {
  if (typeof key !== "string") {
    throw new Error(`Metadata key ${key} must be a string`);

    // If it's an object, throw an error.
  } else if (isAnObject(value)) {
    throw new Error(`Metadata value for ${key} cannot be an object`);

    // If it's an array, make sure there are no objects in it.
  } else if (Array.isArray(value)) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    value.forEach((arrayValue: any) => {
      if (isAnObject(arrayValue)) {
        throw new Error(`Metadata value for member of ${key} cannot be an object`);
      }
    });
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function isAnObject(value: any): boolean {
  return typeof value === "object" && !Array.isArray(value);
}