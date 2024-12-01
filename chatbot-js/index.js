var http = require('http');
var url = require('url');
const { Server } = require("socket.io");
var fs = require('fs');
var NodePy = require('nodepy');

const APName = "Interface for PYTorch";
const APVersion = "Version 0.1.0";
const APAuthor = "By Roberto Rodriguez";
const PORT = process.env.PORT || 8080;



const httpServer = http
  .createServer((request, response) => {
    const { headers, method, url } = request;
    let body = [];
    if (request.method === 'POST' && request.url === '/echo') {
      request
        .on('error', err => {
          console.error(err);
		 // Handle error...
		response.statusCode = 400;
		response.end('400: Bad Request');
        })
        .on('data', chunk => {
          body.push(chunk);
        })
        .on('end', () => {
          body = Buffer.concat(body).toString();
          response.end(body);
        });
    } else if (request.method === 'GET' && request.url === '/echo') {
    request
      .on('error', err => {
        console.error(err);
		 // Handle error...
		response.statusCode = 400;
		response.end('400: Bad Request');
      })
      .on('data', chunk => {
        body.push(chunk);
      })
      .on('end', () => {
        body = Buffer.concat(body).toString();
        // BEGINNING OF NEW STUFF
        response.on('error', err => {
          console.error(err);
        });
        response.statusCode = 200;
        response.setHeader('Content-Type', 'application/json');
        // Note: the 2 lines above could be replaced with this next one:
        // response.writeHead(200, {'Content-Type': 'application/json'})
        const responseBody = { headers, method, url, body };
        response.write(JSON.stringify(responseBody));
        response.end();
        // Note: the 2 lines above could be replaced with this next one:
        // response.end(JSON.stringify(responseBody))
        // END OF NEW STUFF
      });
    } else if (request.method === 'GET' && request.url == '/')  {
        console.log("Detected /");
    request
      .on('error', err => {
        console.error(err);
        response.statusCode = 400;
        response.end('400: Bad Request');
	   
      })
      .on('data', chunk => {
        body.push(chunk);
      })
      .on('end', () => {
        body = Buffer.concat(body).toString();
        response.on('error', err => {
          console.error(err);
        });
        fs.readFile("./chatbot-js/public/index.html", 'utf8', (err, data) => {
          if (!err) {
            console.log("loaded ./public/index.html");
          }
			
            response.setHeader("Content-Type", "text/html");
            response.writeHead(200);
            response.end(data);
		});
      });
    response
      .on('error', err => {
      console.error(err);
      });
    } else if (request.method === 'GET' && request.url == '/favicon.ico')  {
        console.log("Detected /");
    request
      .on('error', err => {
        console.error(err);
        response.statusCode = 400;
        response.end('400: Bad Request');
	   
      })
      .on('data', chunk => {
        body.push(chunk);
      })
      .on('end', () => {
        body = Buffer.concat(body).toString();
        response.on('error', err => {
          console.error(err);
        });
        fs.readFile("./chatbot-js/public/favicon.ico", 'utf8', (err, data) => {
          if (!err) {
            console.log("loaded ./public/favicon.ico");
          }
			
            response.setHeader("Content-Type", "image/vnd.microsoft.icon");
            response.writeHead(200);
            response.end(data);
		});
      });
    response
      .on('error', err => {
      console.error(err);
      });
    } else {
        console.log("Unknow detected "+request.url);
    request
      .on('error', err => {
        console.error(err);
        response.statusCode = 400;
        response.end('400: Bad Request');
	   
      })
      .on('data', chunk => {
        body.push(chunk);
      })
      .on('end', () => {
        body = Buffer.concat(body).toString();
        response.on('error', err => {
          console.error(err);
        });
        fs.readFile("./chatbot-js/unauth.html", 'utf8', (err, data) => {
          if (!err) {
            console.log("loaded unauth.html");
          }
			
            response.setHeader("Content-Type", "text/html");
            response.writeHead(200);
            response.end(data);
		});
      });
    response
      .on('error', err => {
      console.error(err);
      });
    }
  });

const io = new Server(httpServer, { /* options */ });
var rooms = [];
// Setup our python process
var pythonWorker = new NodePy('./chatfp.py');
// Listen for echo back from the python process
pythonWorker.on('chat', function(data)
{
    console.log("[node] Got:", data);

    let createdAt = new Date()
    createdAt = createdAt.toLocaleString()
  
    var messtosend = {
        'sender': 'Bot',
        'text': data[1],
        'createdAt': createdAt
    };
    io.in(data[0]).emit("chat", JSON.stringify(messtosend));
    
    
});
//IO Place

io.on('connection', async (socket) => {
    console.log('A user connected');
    var room = "room"+(rooms.length+1);
    rooms.push(room)
    
    //console.log([...socket.rooms].slice(1, ) );
    socket.join(room)
    socket.on('disconnect', () => {
      console.log('User disconnected');
    });

    /*
    var room = "room"+(rooms.length+1);
    rooms.push(room)

    var room = "room"+(rooms.length+1);
    rooms.push(room)

    var room = "room"+(rooms.length+1);
    rooms.push(room)
    
    console.log(rooms);

    const index = rooms.indexOf("room2");
    if (index > -1) { // only splice array when item is found
        rooms.splice(index, 1); // 2nd parameter means remove one item only
    }

    // array = [2, 9]
    console.log(rooms);
    */

    socket.on('chat', (msg) => {
      console.log('message: ' + msg);

      
      console.log([...socket.rooms].slice(1, )[0] );
      var message = JSON.parse(msg);
      data = message.text;
      message = [...socket.rooms].slice(1, )[0]+":"+data;
      pythonWorker.emit('chat', message);

    });  
  });

httpServer.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
  });



// Start our python process
pythonWorker.start();

