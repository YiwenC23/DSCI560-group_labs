//* Node and NPM are required to be installed
//* 
//* Run the script with the command: node db.js
const express = require("express");
const mysql = require("mysql");
const path = require("path");
const app = express();
const port = process.env.PORT || 3000;


app.use(express.static(path.join(__dirname, "map_project")));


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


app.get("/wells", (req, res) => {
    con.query("SELECT * FROM well_info", (err, result) => {
        if (err) {
            console.error("Database query error:", err);
            return res.status(500).json({ error: "Database query failed" });
        }
        res.json(result);
    });
});


app.listen(port, () => {
    console.log(`Server running on port ${port}`);
});

