
var createError = require('http-errors');
var express = require('express');
var path = require('path');
var cookieParser = require('cookie-parser');
var logger = require('morgan');
var bodyParser = require('body-parser');
var flash = require('express-flash');
var session = require('express-session');
var mysql = require('mysql2');


var connection = mysql.createConnection({
    host: '127.0.0.1',
    user: 'root',
    password: '',
    database: 'test',
    port: '3306'
});

connection.connect;

var app = express();

// set up ejs view engine 
app.set('views', path.join(__dirname, 'views'));
app.set('view engine', 'ejs');

app.use(logger('dev'));
app.use(express.json());
app.use(express.urlencoded({ extended: false }));
app.use(cookieParser());
app.use(express.static(__dirname + '../public'));

app.use(session({
    secret: '123456catr',
    resave: false,
    saveUninitialized: true,
    cookie: { maxAge: 60000 }
}))

app.use(flash());

/* GET home page, respond by rendering index.ejs */
app.get('/add', function (req, res, next) {
    res.render('index', { title: 'Add' });
});

// this code is executed when a user clicks the form submit button
app.post('/create-item', function (req, res, next) {
    var plucode = req.body.plucode;
    var name = req.body.name;
    var lifespan = req.body.lifespan;
    var calories = req.body.calories;
    console.log(plucode);
    function getRandomInt(max) {
        return Math.floor(Math.random() * max);
    }
    let x = getRandomInt(100)
    var sql = `INSERT INTO product (ProductID, PLUcode, Name, Lifespan, Calories) VALUES ('${name} ${x}','${plucode}','${name}','${lifespan}', '${calories}')`;

    console.log(sql);
    connection.query(sql, function (err, result) {
        if (err) throw err;
        console.log('record inserted');
        req.flash('success', 'Data added successfully!');
        res.redirect('/db');
    });
});


//search



//read
app.get('/db/:keyword', (req, res) => {
    var name = req.body.keyword;
    console.log(name);
    var sql = `select * from product where name like "%${name}%"`;
    connection.query(sql, function (error, result) {
        if (error) {
            res.send('Error in database operation');
        } else {
            console.log('The solution is: ', result);
            //res.send(result);
            res.render('productList', { data: result });
        }
    });
});

//delete
app.get('/delete/:ProductID', function (req, res) {
    console.log('test');
    var id = req.params.ProductID;
    console.log('id= ', id);
    var sql = `delete from product where ProductID= "${id}"`;
    console.log(sql);
    connection.query(sql, function (err, rows) {
        if (err) {
            res.end('Failed:' + err)
        } else {
            res.redirect('/db')
        }
    });
});

//update
app.get('/toUpdate/:ProductID', function (req, res) {
    var id = req.params.ProductID;
    console.log('id= ', id);
    var sql = `Select * from product where ProductID= "${id}"`;
    console.log(sql);
    connection.query(sql, function (err, result) {
        if (err) {
            res.end('Failed:' + err);
        } else {
            console.log(result);
            res.render("update", { data: result });
        }
    });
});
app.post('/toUpdate/update', function (req, res, next) {
    var id = req.body.productid
    var plucode = req.body.plucode;
    var name = req.body.name;
    var lifespan = req.body.lifespan;
    var calories = req.body.calories;
    console.log(id);
    var sql = `Update product Set PLUcode= "${plucode}", Name= "${name}", Lifespan = "${lifespan}",Calories="${calories}" Where ProductID = "${id}"`;

    console.log(sql);
    connection.query(sql, function (err, result) {
        if (err) throw err;
        console.log('record update');
        req.flash('success', 'Data added successfully!');
        res.redirect('/db');
    });
});



// catch 404 and forward to error handler
app.use(function (req, res, next) {
    next(createError(404));
});

// error handler
app.use(function (err, req, res, next) {
    // set locals, only providing error in development
    res.locals.message = err.message;
    res.locals.error = req.app.get('env') === 'development' ? err : {};

    // render the error page
    res.status(err.status || 500);
    res.render('error');
});

var http = require('http').Server(app);
var port = 3000;
http.listen(port, function () {
    console.log('Listening to port 3000');
})
/*
// port must be set to 3000 because incoming http requests are routed from port 80 to port 8$
app.listen(3000, function () {
   console.log('Node app is running on port 3000');
});
*/
module.exports = app;
