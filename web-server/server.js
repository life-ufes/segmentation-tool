const express = require('express');
const path = require('path');
const cors = require('cors'); // Import the cors middleware
const app = express();

// Enable CORS for all routes
app.use(cors());

// Set the view engine to EJS
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));

// Serve static files from the "public" directory
app.use(express.static(path.join(__dirname, 'public')));

// Load environment variables from .env file
require('dotenv').config();

// Get the API URL from the environment variable
const apiUrl = process.env.API_URL || 'http://127.0.0.1:5000';
console.log("API URL: ", apiUrl);

app.get('/', function(req, res) {
    res.render('index', { apiUrl });
});

app.get('/index', function(req, res) {
    res.render('index', { apiUrl });
});

app.get('/manual', function(req, res) {
    res.render('manual', { apiUrl });
});

app.get('/sam', function(req, res) {
    res.render('app_sam', { apiUrl });
});

const port = process.env.PORT || 4000;
app.listen(port, function() {
    console.log(`Server is listening on port ${port}`);
});
