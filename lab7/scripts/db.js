var mysql = require("mysql2");

var con = mysql.createConnection({
    host: "localhost",
    user: "root",
    password: "yiwen",
});

con.connect(function(err) {
    if (err) throw err;
    console.log("Connected!");
});

con.query("SELECT * FROM well_info", function (err, result) {
    if (err) throw err;
    console.log(result);
});
