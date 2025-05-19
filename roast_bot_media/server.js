import express from 'express';
import bodyParser from 'body-parser';
import fs from 'fs';
import path from 'path';
import { createObjectCsvWriter as csvWriter } from 'csv-writer';

const app         = express();
const PORT        = 3000;

// Sessions live in the root directory
const rootDir     = process.cwd();
const sessionsDir = rootDir;
const csvPath     = path.join(rootDir, 'data.csv');

app.use(bodyParser.json());
app.use(express.static('public'));

// Serve session folders directly (e.g., /media/session_xxx/image.jpg)
app.use('/media', express.static(sessionsDir));
app.use('/logo.jpeg', express.static(path.join(rootDir, 'logo.jpeg')));
app.use('/logo2.png',     express.static(path.join(rootDir,'logo2.png')));
app.use('/Husky_A300.png',express.static(path.join(rootDir,'Husky_A300.png')));



function newestSessionName() {
  return fs.readdirSync(sessionsDir)
           .filter(n => n.startsWith('session_'))
           .sort().reverse()[0] || '';
}
function listImages(name) {
  if (!name) return [];
  const dir = path.join(sessionsDir, name);
  if (!fs.existsSync(dir)) return [];
  return fs.readdirSync(dir)
           .filter(f => /\.(png|jpe?g|gif)$/i.test(f))
           .map(f => `/media/${name}/${f}`);
}

const writer = csvWriter({
  path: csvPath,
  header: [
    { id: 'firstName',    title: 'FirstName' },
    { id: 'lastName',     title: 'LastName'  },
    { id: 'company',      title: 'Company'   },
    { id: 'position',     title: 'Position'  },
    { id: 'email',        title: 'Email'     },
    { id: 'sessionName',  title: 'Session'   }
  ],
  append: fs.existsSync(csvPath)
});

app.post('/submit', async (req, res) => {
  const sessionName = req.body.sessionName || newestSessionName();
  await writer.writeRecords([{ ...req.body, sessionName }]);
  res.json({ ok: true, sessionName });
});

app.get('/api/sessions', (_req, res) => {
  const names = fs.readdirSync(sessionsDir)
                  .filter(n => n.startsWith('session_'))
                  .sort().reverse();
  res.json(names);
});

app.get('/api/session/:name/images', (req, res) => {
  res.json(listImages(req.params.name));
});

app.listen(PORT, () =>
  console.log(`✓ Roast logger running at: http://localhost:${PORT}`)
);
