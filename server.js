const express = require('express');
const path = require('path');
const app = express();

// Serve static files from the "public" directory
app.use(express.static(path.join(__dirname, 'public')));


app.get('/', function(req, res) {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.get('/manual', function(req, res) {
    res.sendFile(path.join(__dirname, 'public', 'manual.html'));
});


app.get('/sam-click', function(req, res) {
    res.sendFile(path.join(__dirname, 'public', 'app_with_SAM.html'));
});

app.get('/sam-box', function(req, res) {
    res.sendFile(path.join(__dirname, 'public', 'app_with_box.html'));
});

const port = process.env.PORT || 4002;
app.listen(port, function() {
    
    console.log(`Server is listening on port ${port}`);
});