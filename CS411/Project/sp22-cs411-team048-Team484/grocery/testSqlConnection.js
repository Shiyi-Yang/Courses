const mysql = require('mysql');

const options = {
  host:'127.0.0.1',
  user:'root',
  password:'',
  database:'test',
  port:'3306'
};

const connection = mysql.createConnection(options);

connection.connect((err)=>{
  if(err) return console.log(err);
  console.log('[Mysql connect]');
});

const userSelectSql = 'select * from student';
connection.query(userSelectSql,function (err, result) {
    if(err){
      console.log('[SELECT ERROR] - ',err.message);
      return;
    }
   console.log(result);
});

connection.end((err)=>{
  if(err) return console.log(err);
});