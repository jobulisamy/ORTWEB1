import React, { ReactNode } from 'react';
import { Button, Container, Grid, Link, TextField, Switch, FormControlLabel, Table, TableHead, TableBody, TableRow, TableCell, TableContainer } from '@mui/material';
import './App.css';
import Plot from 'react-plotly.js';
import * as ort from 'onnxruntime-web/training';
import { ImageData } from './data';

const App: React.FC = () => {
    // Constants
    const batchSize = 32;
    const numEpochs = 5;
    const lossNodeName = "onnx::loss::8";
    const logIntervalMs = 1000;
    const waitAfterLoggingMs = 500;
    let lastLogTime = 0;
    let messagesQueue: string[] = [];

    // State hooks
    const [maxNumTrainSamples, setMaxNumTrainSamples] = React.useState<number>(293);
    const [maxNumTestSamples, setMaxNumTestSamples] = React.useState<number>(73);
    const [dataFolderPath, setDataFolderPath] = React.useState<string>("");
    const [trainingLosses, setTrainingLosses] = React.useState<number[]>([]);
    const [testAccuracies, setTestAccuracies] = React.useState<number[]>([]);
    const [isTraining, setIsTraining] = React.useState<boolean>(false);
    const [enableLiveLogging, setEnableLiveLogging] = React.useState<boolean>(false);
    const [statusMessage, setStatusMessage] = React.useState<string>("");
    const [errorMessage, setErrorMessage] = React.useState<string>("");
    const [messages, setMessages] = React.useState<string[]>([]);
    const [moreInfoIsCollapsed, setMoreInfoIsCollapsed] = React.useState<boolean>(true);

    // Logging helpers
    function toggleMoreInfoIsCollapsed() {
        setMoreInfoIsCollapsed(!moreInfoIsCollapsed);
    }

    async function logMessage(message: string) {
        messagesQueue.push(message);
        if (Date.now() - lastLogTime > logIntervalMs) {
            setStatusMessage(message);
            if (enableLiveLogging) {
                setMessages(messages => [...messages, ...messagesQueue]);
                messagesQueue = [];
            }
            await new Promise(r => setTimeout(r, waitAfterLoggingMs));
            lastLogTime = Date.now();
        }
    }

    function clearOutputs() {
        setTrainingLosses([]);
        setTestAccuracies([]);
        setMessages([]);
        setStatusMessage("");
        setErrorMessage("");
        messagesQueue = [];
    }

    // Example list of imagePaths
    const imagePaths: string[] = [
        'path/to/image1.jpg',
        'path/to/image2.jpg',
    ];

    // Training and testing functions
    async function runTrainingEpoch(session: ort.TrainingSession, dataSet: ImageData, epoch: number) {
        let batchNum = 0;
        let totalNumBatches = dataSet.getNumTrainingBatches();
        const epochStartTime = Date.now();
        let iterationsPerSecond = 0;
        await logMessage(`TRAINING | Epoch: ${String(epoch + 1).padStart(2)} / ${numEpochs} | Starting training...`);
        for await (const batch of dataSet.trainingBatches()) {
            ++batchNum;
            const feeds = {
                input: batch.data,
                labels: batch.labels
            };
            const results = await session.runTrainStep(feeds);
            const loss = parseFloat(results[lossNodeName].data as any);
            setTrainingLosses(losses => losses.concat(loss));
            iterationsPerSecond = batchNum / ((Date.now() - epochStartTime) / 1000);
            const message = `TRAINING | Epoch: ${String(epoch + 1).padStart(2)} | Batch ${String(batchNum).padStart(3)} / ${totalNumBatches} | Loss: ${loss.toFixed(4)} | ${iterationsPerSecond.toFixed(2)} it/s`;
            await logMessage(message);
            await session.runOptimizerStep();
            await session.lazyResetGrad();
        }
        return iterationsPerSecond;
    }

    async function runTestingEpoch(session: ort.TrainingSession, dataSet: ImageData, epoch: number): Promise<number> {
        let batchNum = 0;
        let totalNumBatches = dataSet.getNumTestBatches();
        let numCorrect = 0;
        let testPicsSoFar = 0;
        let accumulatedLoss = 0;
        const epochStartTime = Date.now();
        await logMessage(`TESTING | Epoch: ${String(epoch + 1).padStart(2)} / ${numEpochs} | Starting testing...`);
        for await (const batch of dataSet.testBatches()) {
            ++batchNum;
            const feeds = {
                input: batch.data,
                labels: batch.labels
            };
            const results = await session.runEvalStep(feeds);
            const loss = parseFloat(results[lossNodeName].data as any);
            accumulatedLoss += loss;
            testPicsSoFar += batch.data.dims[0];
            numCorrect += countCorrectPredictions(results['output'], batch.labels);
            const iterationsPerSecond = batchNum / ((Date.now() - epochStartTime) / 1000);
            const message = `TESTING | Epoch: ${String(epoch + 1).padStart(2)} | Batch ${String(batchNum).padStart(3)} / ${totalNumBatches} | Average test loss: ${(accumulatedLoss / batchNum).toFixed(2)} | Accuracy: ${numCorrect}/${testPicsSoFar} (${(100 * numCorrect / testPicsSoFar).toFixed(2)}%) | ${iterationsPerSecond.toFixed(2)} it/s`;
            await logMessage(message);
        }
        const avgAcc = numCorrect / testPicsSoFar;
        setTestAccuracies(accs => accs.concat(avgAcc));
        return avgAcc;
    }

    async function train() {
        clearOutputs();
        setIsTraining(true);

        const trainingSession = await loadTrainingSession();
        const dataSet = new ImageData(dataFolderPath);
        lastLogTime = Date.now();
        const startTrainingTime = Date.now();
        setStatusMessage('Training started');
        let itersPerSecCumulative = 0;
        let testAcc = 0;
        for (let epoch = 0; epoch < numEpochs; epoch++) {
            itersPerSecCumulative += await runTrainingEpoch(trainingSession, dataSet, epoch);
            testAcc = await runTestingEpoch(trainingSession, dataSet, epoch);
        }
        const trainingTimeMs = Date.now() - startTrainingTime;
        setStatusMessage(`Training completed. Final test set accuracy: ${(100 * testAcc).toFixed(2)}% | Total training time: ${trainingTimeMs / 1000} seconds | Average iterations / second: ${(itersPerSecCumulative / numEpochs).toFixed(2)}`);
        setIsTraining(false);
    }

    async function loadTrainingSession(): Promise<ort.TrainingSession> {
        setStatusMessage('Attempting to load training session...');
        const chkptPath = 'checkpoint';
        const trainingPath = 'training_model.onnx';
        const optimizerPath = 'optimizer_model.onnx';
        const evalPath = 'eval_model.onnx';
        const createOptions: ort.TrainingSessionCreateOptions = {
            checkpointState: chkptPath,
            trainModel: trainingPath,
            evalModel: evalPath,
            optimizerModel: optimizerPath
        };
        try {
            const session = await ort.TrainingSession.create(createOptions);
            setStatusMessage('Training session loaded');
            return session;
        } catch (err) {
            setErrorMessage('Error loading the training session: ' + err);
            console.log("Error loading the training session: ", err);
            console.log("Checkpoint Path: ", chkptPath);
            console.log("Training Model Path: ", trainingPath);
            console.log("Optimizer Model Path: ", optimizerPath);
            console.log("Eval Model Path: ", evalPath);
            throw err;
        }
    }

    // Utility functions
    function countCorrectPredictions(output: ort.Tensor, labels: ort.Tensor): number {
        let result = 0;
        const predictions = getPredictions(output);
        for (let i = 0; i < predictions.length; ++i) {
            if (predictions[i] === labels.data[i]) {
                ++result;
            }
        }
        return result;
    }

    function getPredictions(results: ort.Tensor): number[] {
        const predictions = [];
        const [batchSize, numClasses] = results.dims;
        for (let i = 0; i < batchSize; ++i) {
            const probabilities = results.data.slice(i * numClasses, (i + 1) * numClasses) as Float32Array;
            const resultsLabel = indexOfMax(probabilities);
            predictions.push(resultsLabel);
        }
        return predictions;
    }

    function indexOfMax(arr: Float32Array): number {
        if (arr.length === 0) {
            throw new Error('index of max (used in test accuracy function) expects a non-empty array. Something went wrong.');
        }
        let maxIndex = 0;
        for (let i = 1; i < arr.length; i++) {
            if (arr[i] > arr[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    return (
        <Container>
            <h1>ONNX Runtime Web Training</h1>
            <Grid container spacing={3}>
                <Grid item xs={12}>
                    <TextField
                        label="Data Folder Path"
                        value={dataFolderPath}
                        onChange={(e) => setDataFolderPath(e.target.value)}
                        fullWidth
                    />
                </Grid>
                <Grid item xs={6}>
                    <TextField
                        label="Max Number of Training Samples"
                        type="number"
                        value={maxNumTrainSamples}
                        onChange={(e) => setMaxNumTrainSamples(Number(e.target.value))}
                        fullWidth
                    />
                </Grid>
                <Grid item xs={6}>
                    <TextField
                        label="Max Number of Testing Samples"
                        type="number"
                        value={maxNumTestSamples}
                        onChange={(e) => setMaxNumTestSamples(Number(e.target.value))}
                        fullWidth
                    />
                </Grid>
                <Grid item xs={6}>
                    <FormControlLabel
                        control={<Switch checked={enableLiveLogging} onChange={(e) => setEnableLiveLogging(e.target.checked)} />}
                        label="Enable Live Logging"
                    />
                </Grid>
                <Grid item xs={12}>
                    <Button variant="contained" color="primary" onClick={train} disabled={isTraining}>
                        {isTraining ? 'Training...' : 'Start Training'}
                    </Button>
                </Grid>
                <Grid item xs={12}>
                    <h2>Status</h2>
                    <p>{statusMessage}</p>
                    {errorMessage && <p style={{ color: 'red' }}>{errorMessage}</p>}
                </Grid>
                <Grid item xs={12}>
                    <h2>Logs</h2>
                    <div style={{ maxHeight: '200px', overflowY: 'scroll' }}>
                        {messages.map((msg, idx) => (
                            <p key={idx}>{msg}</p>
                        ))}
                    </div>
                </Grid>
                <Grid item xs={12}>
                    <h2>Training Losses</h2>
                    <Plot
                        data={[
                            {
                                x: trainingLosses.map((_, i) => i),
                                y: trainingLosses,
                                type: 'scatter',
                                mode: 'lines+markers',
                                marker: { color: 'red' },
                            },
                        ]}
                        layout={{ width: 720, height: 440, title: 'Training Loss' }}
                    />
                </Grid>
                <Grid item xs={12}>
                    <h2>Test Accuracies</h2>
                    <Plot
                        data={[
                            {
                                x: testAccuracies.map((_, i) => i),
                                y: testAccuracies,
                                type: 'scatter',
                                mode: 'lines+markers',
                                marker: { color: 'blue' },
                            },
                        ]}
                        layout={{ width: 720, height: 440, title: 'Test Accuracy' }}
                    />
                </Grid>
            </Grid>
        </Container>
    );
};

export default App;
