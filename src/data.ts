import * as ort from 'onnxruntime-web';

export class ImageData {
    private batchSize: number = 32; // Adjust as needed
    private maxNumTrainSamples: number = 293; // Adjust as needed
    private maxNumTestSamples: number = 73; // Adjust as needed
    private dataFolderPath: string = "public/data"; // Assuming images are in 'public/data/train' and 'public/data/test'

    constructor(dataFolderPath: string) {
        this.dataFolderPath = dataFolderPath;
    }

    public async prepareData() {
        // No need to create labels.txt files anymore
    }

    public async * trainingBatches() {
        const trainingData = await this.loadData('train', this.maxNumTrainSamples);
        const trainingLabels = await this.loadLabels('train', this.maxNumTrainSamples);
        yield* this.batches(trainingData, trainingLabels);
    }

    public async * testBatches() {
        const testData = await this.loadData('test', this.maxNumTestSamples);
        const testLabels = await this.loadLabels('test', this.maxNumTestSamples);
        yield* this.batches(testData, testLabels);
    }

    private async loadData(folderName: string, maxSamples: number): Promise<ort.Tensor[]> {
        console.debug(`Loading images from "${folderName}".`);
        const files = await this.loadFilesInFolder(folderName);

        const result: ort.Tensor[] = [];
        for (let i = 0; i < files.length && i < maxSamples; i++) {
            const imagePath = `${this.dataFolderPath}/${folderName}/${files[i]}`;
            const image = await this.loadImage(imagePath); // Load image data
            const imageData = await this.processImage(image); // Process image data
            result.push(imageData);
        }

        return result;
    }

    private async loadLabels(folderName: string, maxSamples: number): Promise<ort.Tensor[]> {
        console.debug(`Loading labels for "${folderName}".`);
        const labelsFilePath = `${this.dataFolderPath}/${folderName}/${folderName}_labels.txt`;
        const labelsData = await this.readLabelsFile(labelsFilePath, maxSamples);
        return labelsData;
    }

    private async readLabelsFile(labelsFilePath: string, maxSamples: number): Promise<ort.Tensor[]> {
        const response = await fetch(labelsFilePath);
        const text = await response.text();
        const labels = text.split('\n').map(label => parseInt(label.trim())).filter(label => !isNaN(label));
        
        const labelsData = labels.slice(0, maxSamples).map(label => new ort.Tensor('int64', new BigInt64Array([BigInt(label)]), [1]));
        return labelsData;
    }

    private async loadImage(imagePath: string): Promise<Uint8Array> {
        console.debug(`Loading image from "${imagePath}".`);
        const response = await fetch(imagePath);
        if (!response.ok) {
            throw new Error(`Failed to fetch image ${imagePath}.`);
        }
        const imageData = await response.arrayBuffer();
        return new Uint8Array(imageData);
    }

    private async processImage(imageData: Uint8Array): Promise<ort.Tensor> {
        console.debug(`Processing image data.`);
        // Example: Convert image data to tensor (replace with your image processing logic)
        const imageTensor = new ort.Tensor('float32', new Float32Array(imageData), [1, imageData.length]);
        return imageTensor;
    }

    private *batches(data: ort.Tensor[], labels: ort.Tensor[]): IterableIterator<{ data: ort.Tensor; labels: ort.Tensor; }> {
        const numBatches = Math.floor(data.length / this.batchSize);
        for (let i = 0; i < numBatches; i++) {
            const startIndex = i * this.batchSize;
            const batchData = data.slice(startIndex, startIndex + this.batchSize);
            const batchLabels = labels.slice(startIndex, startIndex + this.batchSize);
            yield { data: batchData[0], labels: batchLabels[0] };
        }
    }

    public static normalize(pixelValue: number): number {
        return ((pixelValue / 255) - 0.5) / 0.5; // Example normalization function
    }

    private async loadFilesInFolder(folderName: string): Promise<string[]> {
        console.debug(`Loading files from "${folderName}".`);
        const response = await fetch(`${this.dataFolderPath}/${folderName}`);
        const text = await response.text();
        const files = text.split('\n').filter(file => !!file);
        return files;
    }
}
