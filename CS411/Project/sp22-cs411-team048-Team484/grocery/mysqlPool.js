const mysql = require('mysql');

class MysqlPool {
  constructor(){
    this.flag = true;
    this.pool = mysql.createPool({
      host:'127.0.0.1',
      user:'root',
      password:'',
      database:'test',
      port:3306
    });
  }
  getPool(){
    if(this.flag){
      this.pool.on('connection', (connection)=>{
        connection.query('SET SESSION auto_increment_increment=1');
        this.flag = false;
      });
    }
    return this.pool;
  }  
}

module.exports = MysqlPool;