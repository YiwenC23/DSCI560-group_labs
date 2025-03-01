//* Node and NPM are required to be installed
//* 
//* Run the script with the command: node db.js
const express = require("express");
const mysql = require("mysql");
const app = express();
const port = 3000;


app.use(express.static("map_project"));


var con = mysql.createConnection({
    host: "localhost",
    user: "root",
    password: "yiwen",
    database: "lab7"
});


con.connect(function(err) {
    if (err) throw err;
    console.log("Successfully connected to the database!");
});


app.get("/wells", function(req, res) {
    con.query("SELECT * FROM well_info", function(err, result){
        if(err) throw err;
        res.json(result);
    });
});


app.listen(port, () => console.log(`Server running on port ${port}`));