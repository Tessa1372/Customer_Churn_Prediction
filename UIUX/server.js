// server.js
const express = require('express');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Serve index.html as the homepage
app.get('/', (req, res) => {
    res.sendFile(__dirname + '/index.html');
});
// Endpoint to handle prediction request
app.post('/predict', async (req, res) => {
    try {
        // Extract feature values from request body
        const { accountWeeks, dataUsage } = req.body;
        // Extract other feature values similarly

        // Make a POST request to ModelBit endpoint
        const modelBitResponse = await axios.post('https://tessasmampilly.ap-south-1.modelbit.com/v1/predict/latest', {
            data: {
                AccountWeeks: accountWeeks,
                DataUsage: dataUsage
                // Add other feature values similarly
            }
        });

        // Log the response from ModelBit endpoint
        console.log('ModelBit Response:', modelBitResponse.data);

        // Extract prediction from ModelBit response
        let prediction = modelBitResponse.data.prediction;
        
        // If prediction is undefined, default to 0 (negative churn)
        if (typeof prediction === 'undefined') {
            console.log('Warning: Prediction is undefined. Defaulting to negative churn (0).');
            prediction = 0;
        }

        // Log the final prediction
        console.log('Prediction:', prediction);

        // Send prediction back to client
        res.json({ prediction });
    } catch (error) {
        console.error('Prediction error:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});


// Start the server
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
