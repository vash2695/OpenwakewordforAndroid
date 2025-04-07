package com.example.openwakeword;
import android.content.res.AssetManager;
import android.util.Log;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;


class ONNXModelRunner implements AutoCloseable {

    private static final int BATCH_SIZE = 1; // Replace with your batch size

    private final AssetManager assetManager;
    private final OrtEnvironment env;
    private final OrtSession melspectrogramSession;
    private final OrtSession embeddingSession;
    private final OrtSession heyNuggetSession;

    public ONNXModelRunner(AssetManager assetManager) throws IOException, OrtException {
        this.assetManager = assetManager;
        this.env = OrtEnvironment.getEnvironment(); // Get the environment once

        // Load all models and create sessions in the constructor
        melspectrogramSession = env.createSession(readModelFile("melspectrogram.onnx"));
        embeddingSession = env.createSession(readModelFile("embedding_model.onnx"));
        heyNuggetSession = env.createSession(readModelFile("hey_nugget_new.onnx"));
    }

    public float[][] get_mel_spectrogram(float[] inputArray) throws OrtException {
        float[][] outputArray = null;
        int SAMPLES = inputArray.length;
        OnnxTensor inputTensor = null; // Declare outside try for finally block

        try {
            // Convert the input array to ONNX Tensor
            FloatBuffer floatBuffer = FloatBuffer.wrap(inputArray);
            // Use the shared environment 'env'
            inputTensor = OnnxTensor.createTensor(env, floatBuffer, new long[]{BATCH_SIZE, SAMPLES});

            // Run the model using the member session
            try (OrtSession.Result results = melspectrogramSession.run(Collections.singletonMap(melspectrogramSession.getInputNames().iterator().next(), inputTensor))) {
                float[][][][] outputTensor = (float[][][][]) results.get(0).getValue();
                float[][] squeezed = squeeze(outputTensor);
                outputArray = applyMelSpecTransform(squeezed);
            }
        } catch (Exception e) {
            // Log the exception properly instead of just printing stack trace
             Log.e("ONNXModelRunner", "Error during mel spectrogram inference", e);
             // Optionally re-throw or handle differently
             throw new OrtException("Mel spectrogram inference failed. See logs for details.");
        } finally {
            if (inputTensor != null) inputTensor.close();
            // Do NOT close the session here, it's managed by the class
            // if (session!=null) session.close(); // Remove
        }
        // OrtEnvironment.getEnvironment().close(); // REMOVE: Environment should not be closed here
        return outputArray;
    }
    public static float[][] squeeze(float[][][][] originalArray) {
        float[][] squeezedArray = new float[originalArray[0][0].length][originalArray[0][0][0].length];
        for (int i = 0; i < originalArray[0][0].length; i++) {
            for (int j = 0; j < originalArray[0][0][0].length; j++) {
                squeezedArray[i][j] = originalArray[0][0][i][j];
            }
        }

        return squeezedArray;
    }
    public static float[][] applyMelSpecTransform(float[][] array) {
        float[][] transformedArray = new float[array.length][array[0].length];

        for (int i = 0; i < array.length; i++) {
            for (int j = 0; j < array[i].length; j++) {
                transformedArray[i][j] = array[i][j] / 10.0f + 2.0f;
            }
        }

        return transformedArray;
    }

    public float[][] generateEmbeddings(float[][][][] input) throws OrtException {
       // OrtEnvironment env = OrtEnvironment.getEnvironment(); // Remove: Use member variable env
       // InputStream is = assetManager.open("embedding_model.onnx"); // Remove: Session loading moved to constructor
       // byte[] model = new byte[is.available()]; // Remove
       // is.read(model); // Remove
       // is.close(); // Remove
       // OrtSession sess = env.createSession(model); // Remove: Use member session

        OnnxTensor inputTensor = null; // Declare outside try for finally block

        try {
            // Use the shared environment 'env'
            inputTensor = OnnxTensor.createTensor(env, input);
            // Use the member session 'embeddingSession'
            try (OrtSession.Result results = embeddingSession.run(Collections.singletonMap("input_1", inputTensor))) { // Assuming input name is "input_1"
                // Extract the output tensor
                float[][][][] rawOutput = (float[][][][]) results.get(0).getValue();

                // Assuming the output shape is (N, 1, 1, 96), and we want to reshape it to (N, 96)
                // Check dimensions before accessing
                 if (rawOutput == null || rawOutput.length == 0 || rawOutput[0].length == 0 || rawOutput[0][0].length == 0 || rawOutput[0][0][0].length == 0) {
                    throw new OrtException("Embedding model produced invalid output shape.");
                 }
                float[][] reshapedOutput = new float[rawOutput.length][rawOutput[0][0][0].length];
                for (int i = 0; i < rawOutput.length; i++) {
                   // Ensure inner dimensions are also valid if necessary, depending on model guarantees
                   if (rawOutput[i].length > 0 && rawOutput[i][0].length > 0 && rawOutput[i][0][0].length > 0) {
                        System.arraycopy(rawOutput[i][0][0], 0, reshapedOutput[i], 0, rawOutput[i][0][0].length);
                    } else {
                         // Handle potential inconsistent inner dimensions if the model doesn't guarantee uniformity
                        Log.w("ONNXModelRunner", "Inconsistent inner dimension in embedding output at index " + i);
                        // Depending on requirements, you might fill with zeros, skip, or throw an error
                    }
                }
                return reshapedOutput;
            }
        } catch (Exception e) {
            Log.e("ONNXModelRunner", "Error during embedding generation", e);
             // Re-throw as OrtException or handle appropriately
             throw new OrtException("Embedding generation failed. See logs for details.");
            // return null; // Avoid returning null, prefer exceptions for errors
        } finally {
            if (inputTensor != null) inputTensor.close();
            // if (sess != null) sess.close(); // REMOVE: Session is now managed by the class
        }
       // env.close(); // REMOVE: Environment should not be closed here
       // Return null removed, exception is thrown instead
    }


    public String predictWakeWord(float[][][] inputArray) throws OrtException {
        float[][] result; // No need to initialize here
        String resultant = ""; // Initialize as empty

        OnnxTensor inputTensor = null; // Declare outside try for finally block

        try {
            // Create a tensor from the input array using the shared environment 'env'
            inputTensor = OnnxTensor.createTensor(env, inputArray);
            // Run the inference using the member session 'heyNuggetSession'
            OrtSession.Result outputs = heyNuggetSession.run(Collections.singletonMap(heyNuggetSession.getInputNames().iterator().next(), inputTensor));
            // Extract the output tensor, convert it to the desired type
            result = (float[][]) outputs.get(0).getValue();
             // Check if result is valid before accessing
             if (result != null && result.length > 0 && result[0].length > 0) {
                resultant = String.format("%.5f", (double) result[0][0]);
             } else {
                 Log.e("ONNXModelRunner", "Wake word model produced invalid output.");
                 // Handle error appropriately, maybe return a default value or throw
                 resultant = "Error"; // Or throw new OrtException("Invalid wake word output");
             }

        } catch (OrtException e) {
            Log.e("ONNXModelRunner", "Error during wake word prediction", e);
            throw e; // Re-throw the exception
        } finally {
            if (inputTensor != null) inputTensor.close();
            // Do NOT close the session here
        }
        return resultant;
    }

    private byte[] readModelFile(String filename) throws IOException {
        try (InputStream is = assetManager.open(filename)) {
            // Efficiently read bytes
            ByteArrayOutputStream buffer = new ByteArrayOutputStream();
            int nRead;
            byte[] data = new byte[16384]; // Or another reasonable buffer size
            while ((nRead = is.read(data, 0, data.length)) != -1) {
                buffer.write(data, 0, nRead);
            }
            return buffer.toByteArray();
        }
    }

     // Implement AutoCloseable to release resources
     @Override
     public void close() throws OrtException {
         // Close sessions in reverse order of creation (optional, but good practice)
         if (heyNuggetSession != null) heyNuggetSession.close();
         if (embeddingSession != null) embeddingSession.close();
         if (melspectrogramSession != null) melspectrogramSession.close();
         // Close the environment last
         if (env != null) env.close();
         Log.i("ONNXModelRunner", "ONNX resources closed.");
     }
}


public class Model {
    int n_prepared_samples=1280;
    int sampleRate=16000;
    int melspectrogramMaxLen= 10*97;
    int feature_buffer_max_len=120;
    ONNXModelRunner modelRunner;
    float[][] featureBuffer;
    ArrayDeque<Float> raw_data_buffer=new ArrayDeque<>(sampleRate * 10);;
    float[] raw_data_remainder = new float[0];
    float[][] melspectrogramBuffer;
    int accumulated_samples=0;
    Model(ONNXModelRunner modelRunner) {
        melspectrogramBuffer = new float[76][32];
        for (int i = 0; i < melspectrogramBuffer.length; i++) {
            for (int j = 0; j < melspectrogramBuffer[i].length; j++) {
                melspectrogramBuffer[i][j] = 1.0f; // Assign 1.0f to simulate numpy.ones
            }
        }
        this.modelRunner=modelRunner;
        try{

            // Initialize featureBuffer with zeros or a placeholder, not random embeddings
            // The size calculation might need adjustment based on expected embedding output dimensions (e.g., 96)
             int embeddingDim = 96; // Assuming the embedding model outputs 96 features
             this.featureBuffer = new float[feature_buffer_max_len][embeddingDim];
             // Initialize with zeros or a suitable placeholder value if needed
             // for (int i = 0; i < feature_buffer_max_len; i++) {
             //     Arrays.fill(this.featureBuffer[i], 0.0f);
             // }

             // The original random initialization seems incorrect for a real application.
             // this.featureBuffer = this._getEmbeddings(this.generateRandomIntArray(16000 * 4), 76, 8);

        }
    catch (Exception e) // Catch specific exceptions if possible (IOException, OrtException)
    {

        // Log error properly
         Log.e("ModelInit", "Error initializing feature buffer", e);
         // Handle initialization failure, maybe rethrow or set a flag
    }

    }

    public float[][][] getFeatures(int nFeatureFrames, int startNdx) {
        int endNdx;
        if (startNdx != -1) {
            endNdx = (startNdx + nFeatureFrames != 0) ? (startNdx + nFeatureFrames) : featureBuffer.length;
        } else {
            startNdx = Math.max(0, featureBuffer.length - nFeatureFrames); // Ensure startNdx is not negative
            endNdx = featureBuffer.length;
        }

        int length = endNdx - startNdx;
        float[][][] result = new float[1][length][featureBuffer[0].length]; // Assuming the second dimension has fixed size.

        for (int i = 0; i < length; i++) {
            System.arraycopy(featureBuffer[startNdx + i], 0, result[0][i], 0, featureBuffer[startNdx + i].length);
        }

        return result;
    }

    // Java equivalent to _get_embeddings method
    private float[][] _getEmbeddings(float[] x, int windowSize, int stepSize) throws OrtException, IOException {

        // Ensure modelRunner is not null before using
         if (this.modelRunner == null) {
             throw new IllegalStateException("ONNXModelRunner is not initialized.");
         }

        float[][] spec = this.modelRunner.get_mel_spectrogram(x); // Assuming this method exists and returns float[][]

         // Add null check for spec
         if (spec == null || spec.length == 0 || spec[0].length == 0) {
             Log.w("Model", "_getEmbeddings: Mel spectrogram is null or empty.");
             return new float[0][0]; // Return empty array or handle appropriately
         }

        ArrayList<float[][]> windows = new ArrayList<>();
        int specCols = spec[0].length; // Get number of columns once

        for (int i = 0; i <= spec.length - windowSize; i += stepSize) {
            float[][] window = new float[windowSize][specCols]; // Use specCols

            for (int j = 0; j < windowSize; j++) {
                 // Add boundary check for spec[i + j] just in case
                 if (i + j < spec.length) {
                     System.arraycopy(spec[i + j], 0, window[j], 0, specCols);
                 }
            }

            // Check if the window is full-sized (not truncated) - this check might be redundant given the loop condition
            // if (window.length == windowSize) { // This should always be true due to loop condition
                 windows.add(window);
            // }
        }

        if (windows.isEmpty()) {
             Log.w("Model", "_getEmbeddings: No valid windows created from spectrogram.");
             return new float[0][0]; // Or handle as needed
         }

        // Convert ArrayList to array and add the required extra dimension
         int numWindows = windows.size();
         // Assuming specCols represents the correct dimension (e.g., 32)
         float[][][][] batch = new float[numWindows][windowSize][specCols][1];
        for (int i = 0; i < numWindows; i++) {
            float[][] currentWindow = windows.get(i);
            for (int j = 0; j < windowSize; j++) {
                for (int k = 0; k < specCols; k++) {
                     // Add bounds check for currentWindow just in case
                     if (j < currentWindow.length && k < currentWindow[j].length) {
                         batch[i][j][k][0] = currentWindow[j][k]; // Add the extra dimension here
                     }
                }
            }
        }
         // Call generateEmbeddings with the prepared batch
         return this.modelRunner.generateEmbeddings(batch);
    }

    // Utility function to generate random int array, equivalent to np.random.randint
    private float[] generateRandomIntArray(int size) {
        float[] arr = new float[size];
        Random random = new Random();
        for (int i = 0; i < size; i++) {
            arr[i] = (float) random.nextInt(2000) - 1000; // range [-1000, 1000)
        }
        return arr;
    }
    public void bufferRawData(float[] x) { // Change double[] to match your actual data type
        // Check if input x is not null
        if (x != null) {
            // Check if raw_data_buffer has enough space, if not, remove old data
            while (raw_data_buffer.size() + x.length > sampleRate * 10) {
                raw_data_buffer.poll(); // or pollFirst() - removes and returns the first element of this deque
            }
            for (float value : x) {
                raw_data_buffer.offer(value); // or offerLast() - Inserts the specified element at the end of this deque
            }
        }
    }

    public void streamingMelSpectrogram(int n_samples) {
        if (raw_data_buffer.size() < 400) {
            throw new IllegalArgumentException("The number of input frames must be at least 400 samples @ 16kHz (25 ms)!");
        }

        // Converting the last n_samples + 480 (3 * 160) samples from raw_data_buffer to an ArrayList
        float[] tempArray = new float[n_samples + 480]; // 160 * 3 = 480
        Object[] rawDataArray = raw_data_buffer.toArray();
        for (int i = Math.max(0, rawDataArray.length - n_samples - 480); i < rawDataArray.length; i++) {
            tempArray[i - Math.max(0, rawDataArray.length - n_samples - 480)] = (Float) rawDataArray[i];
        }

        // Assuming getMelSpectrogram returns a two-dimensional float array
        float[][] new_mel_spectrogram ;
        try {
            new_mel_spectrogram = modelRunner.get_mel_spectrogram(tempArray);
        } catch (OrtException e) {
            throw new RuntimeException(e);
        }

        // Combine existing melspectrogram_buffer with new_mel_spectrogram
        float[][] combined = new float[this.melspectrogramBuffer.length + new_mel_spectrogram.length][];

        System.arraycopy(this.melspectrogramBuffer, 0, combined, 0, this.melspectrogramBuffer.length);
        System.arraycopy(new_mel_spectrogram, 0, combined, this.melspectrogramBuffer.length, new_mel_spectrogram.length);
        this.melspectrogramBuffer = combined;

        // Trim the melspectrogram_buffer if it exceeds the max length
        if (this.melspectrogramBuffer.length > melspectrogramMaxLen) {
            float[][] trimmed = new float[melspectrogramMaxLen][];
            System.arraycopy(this.melspectrogramBuffer, this.melspectrogramBuffer.length - melspectrogramMaxLen, trimmed, 0, melspectrogramMaxLen);
            this.melspectrogramBuffer = trimmed;
        }

    }

    public int streaming_features(float[] audiobuffer) {
        int processed_samples = 0;
        this.accumulated_samples=0;
        if (raw_data_remainder.length != 0) {
            // Create a new array to hold the result of concatenation
            float[] concatenatedArray = new float[raw_data_remainder.length + audiobuffer.length];

            // Copy elements from raw_data_remainder to the new array
            System.arraycopy(raw_data_remainder, 0, concatenatedArray, 0, raw_data_remainder.length);

            // Copy elements from x to the new array, starting right after the last element of raw_data_remainder
            System.arraycopy(audiobuffer, 0, concatenatedArray, raw_data_remainder.length, audiobuffer.length);

            // Assign the concatenated array back to x
            audiobuffer = concatenatedArray;

            // Reset raw_data_remainder to an empty array
            raw_data_remainder = new float[0];
        }

        if (this.accumulated_samples + audiobuffer.length >= 1280) {
            int remainder = (this.accumulated_samples + audiobuffer.length) % 1280;
            if (remainder != 0) {
                // Create an array for x_even_chunks that excludes the last 'remainder' elements of 'x'
                float[] x_even_chunks = new float[audiobuffer.length - remainder];
                System.arraycopy(audiobuffer, 0, x_even_chunks, 0, audiobuffer.length - remainder);

                // Buffer the even chunks of data
                this.bufferRawData(x_even_chunks);

                // Update accumulated_samples by the length of x_even_chunks
                this.accumulated_samples += x_even_chunks.length;

                // Set raw_data_remainder to the last 'remainder' elements of 'x'
                this.raw_data_remainder = new float[remainder];
                System.arraycopy(audiobuffer, audiobuffer.length - remainder, this.raw_data_remainder, 0, remainder);
            } else if (remainder == 0) {
                // Buffer the entire array 'x'
                this.bufferRawData(audiobuffer);

                // Update accumulated_samples by the length of 'x'
                this.accumulated_samples += audiobuffer.length;

                // Set raw_data_remainder to an empty array
                this.raw_data_remainder = new float[0];
            }
        } else {
            this.accumulated_samples += audiobuffer.length;
            this.bufferRawData(audiobuffer); // Adapt this method according to your class
        }


        if (this.accumulated_samples >= 1280 && this.accumulated_samples % 1280 == 0) {

            this.streamingMelSpectrogram(this.accumulated_samples);

            float[][][][] x = new float[1][76][32][1];

            for (int i = (accumulated_samples / 1280) - 1; i >= 0; i--) {

                int ndx = -8 * i;
                if (ndx == 0) {
                    ndx = melspectrogramBuffer.length;
                }
                // Calculate start and end indices for slicing
                int start = Math.max(0, ndx - 76);
                int end = ndx;

                for (int j = start, k = 0; j < end; j++, k++) {
                    for (int w = 0; w < 32; w++) {
                        x[0][k][w][0] = (float) melspectrogramBuffer[j][w];
                    }
                }
                if (x[0].length== 76)
                {
                    try {
                        float[][] newFeatures=modelRunner.generateEmbeddings(x);
                        if (featureBuffer == null) {
                            featureBuffer = newFeatures;
                        } else {
                            int totalRows = featureBuffer.length + newFeatures.length;
                            int numColumns = featureBuffer[0].length; // Assuming all rows have the same length
                            float[][] updatedBuffer = new float[totalRows][numColumns];

                            // Copy original featureBuffer into updatedBuffer
                            for (int l = 0; l< featureBuffer.length; l++) {
                                System.arraycopy(featureBuffer[l], 0, updatedBuffer[l], 0, featureBuffer[l].length);
                            }

                            // Copy newFeatures into the updatedBuffer, starting after the last original row
                            for (int k = 0; k < newFeatures.length; k++) {
                                System.arraycopy(newFeatures[k], 0, updatedBuffer[k + featureBuffer.length], 0, newFeatures[k].length);
                            }

                            featureBuffer = updatedBuffer;
                        }

                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                }
            }
            processed_samples=this.accumulated_samples;
            this.accumulated_samples=0;

        }
        if (featureBuffer.length > feature_buffer_max_len) {
            float[][] trimmedFeatureBuffer = new float[feature_buffer_max_len][featureBuffer[0].length];

            // Copy the last featureBufferMaxLen rows of featureBuffer into trimmedFeatureBuffer
            for (int i = 0; i < feature_buffer_max_len; i++) {
                trimmedFeatureBuffer[i] = featureBuffer[featureBuffer.length - feature_buffer_max_len + i];
            }

            // Update featureBuffer to point to the new trimmedFeatureBuffer
            featureBuffer = trimmedFeatureBuffer;
        }
        return processed_samples != 0 ? processed_samples : this.accumulated_samples;



    }

    public String predict_WakeWord(float[] audiobuffer){

        n_prepared_samples=this.streaming_features(audiobuffer);
        float[][][] res=this.getFeatures(16,-1);
        String result="";
        try {
            result=modelRunner.predictWakeWord(res);
        } catch (OrtException e) {
            throw new RuntimeException(e);
        }
        return  result;
        }
    }

