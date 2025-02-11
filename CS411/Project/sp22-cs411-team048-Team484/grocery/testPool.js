const MysqlPool = require('./mysqlPool');

const mysqlPool = new MysqlPool();

const pool = mysqlPool.getPool();

var sql = 'SELECT * FROM student';

pool.getConnection((err,connection)=>{
  connection.query(sql,(err,res)=>{
    console.log(res);
  });
  connection.release();
});

var  insert = 'INSERT INTO student VALUES(?,?,?,?)';
var  insertParams = [777, 'try again name','testDept', '7.77CN']
pool.getConnection((err,connection)=>{
    connection.query(insert,insertParams,(err,res)=>{
      console.log(res);
    });
    connection.release();
  });