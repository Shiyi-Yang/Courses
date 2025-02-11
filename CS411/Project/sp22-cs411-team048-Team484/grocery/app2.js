// app.js
// import modules
const express = require('express')
const app = express()
const port = 3000
const exphbs = require('express-handlebars');
var bodyParser = require('body-parser');
var mysql = require('mysql2');


var connection = mysql.createConnection({
    host: '127.0.0.1',
    user: 'root',
    password: '',
    database: 'test',
    port: '3306'
});


// import product.json
const product_list = require('./product.json')

// route setting
app.get('/', (req, res) => {
  res.render('index', { productList: product_list.results })
})
// route setting for show: using params
app.get('/products/:id', (req, res) => {
  console.log(req.params.id)
  let product = product_list.results.find(product => product.id == req.params.id)
  console.log(product)
  res.render('show', { product: product })
})
// route setting for search: query string
app.get('/search', (req, res) => {
  const products = product_list.results.filter(product => product.name.toLowerCase().includes(req.query.keyword.toLowerCase()))
  const keyword = req.query.keyword
  res.render('index', { productList: products, keyword: keyword })
})

// create server
app.listen(port, () => {
  console.log(`server listen to http://localhost:${port}`)
})


// setting template engine
app.engine('handlebars', exphbs.engine({ defaultLayout: "main", extname: '.handlebars' }));
app.set('views', './views')
app.set('view engine', 'handlebars')

// setting static files
app.use(express.static('public'))
