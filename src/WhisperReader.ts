import { exec, execSync } from "child_process";
import { Document, BaseReader } from "llamaindex";
import { WhisperModel, WhisperLanguage, WhisperOutputFormat, WhisperTask, WhisperDevice } from "llamaindex-whisper";
import { promisify } from "util";

const promisedExec = promisify(exec);


export class WhisperReader implements BaseReader {
  model: WhisperModel = WhisperModel.Base;
  temperature: number = 0;
  language: WhisperLanguage = WhisperLanguage.English;
  outputDirectory = ".";
  outputFormat = WhisperOutputFormat.All;
  task: WhisperTask = WhisperTask.Transcribe;
  device: WhisperDevice = WhisperDevice.CUDA;

  constructor(init: Partial<WhisperReader> = {}) {
    if (execSync(`which whisper`).includes("not found")) {
      throw "whisper not found in path. Please `pip install whisper` and add it to your path."
    }
    Object.assign(this, init);
  }

  async loadData(filePath: string, metadata: Record<string, any> = {}): Promise<Document[]> {
    return await this.whisperExec(filePath).then(({ stdout, stderr }) => {
      // `promisedExec` will reject the promise if it actually errors
      //  This includes things like warnings, which we don't care about.
      // For example, not having CUDA will cause a warning that it is
      // falling back to CPU mode. We don't care about that.
      if (stderr && !process.env.JEST_WORKER_ID) {
        console.error(stderr);
      }
      const documents = [new Document({ text: this.cleanWhisperOutput(stdout), metadata })];
      return Promise.resolve(documents);
    });
  }

  async whisperExec(filePath: string): Promise<{ stdout: string, stderr: string }> {
    const whisperCommand = [
      `whisper ${filePath}`,
      `--model ${this.model}`,
      `--temperature ${this.temperature}`,
      `--language ${this.language}`,
      `--output_dir ${this.outputDirectory}`,
      `--output_format ${this.outputFormat}`,
      `--task ${this.task}`,
      `--device ${this.device}`
    ].join(" ");
    console.debug(`Running whisper command:\n  ${whisperCommand}`)
    return promisedExec(whisperCommand);
  }

  cleanWhisperOutput(output: string): string {
    // Whisper outputs information at the beginning that isn't the actual
    // audio transcript.
    // We want everything in output after the start of the transcription,
    // marked by the first timestamp: [00:00.000
    console.log(output);
    const startOfTranscription = output.indexOf('[00:00.000');
    const resultOutput = output.slice(startOfTranscription);
    return resultOutput;
  }


}