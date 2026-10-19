/**
 * Williams Dining Hall Wait-Time Logger — COMP 365 Individual Project
 * Google Apps Script web app. Anonymous: stores no names, IDs, emails or device info.
 *
 * Setup:
 *  1. Create a Google Sheet. Extensions > Apps Script.
 *  2. Paste Code.gs and Index.html (File + > HTML, name it "Index").
 *     Project Settings > check "Show appsscript.json" and paste appsscript.json.
 *  3. Run setup() once (authorize when prompted) to create the header row.
 *  4. Deploy > New deployment > Web app. Execute as: Me. Who has access: Anyone.
 *  5. Share the /exec URL (or a QR code of it) at the Williams entrance.
 *  6. File > Download > CSV to get data for the notebook.
 */

const SHEET_NAME = 'log';
const HEADERS = [
  'submission_id', 'source', 'line_join_time', 'card_tap_time',
  'visual_queue_category', 'actual_wait_minutes', 'server_received_time', 'flag'
];
const QUEUE_CATEGORIES = [
  'A: Inside turnstiles (0-5 people)',
  'B: Back to the double doors (6-15 people)',
  'C: Past double doors into lobby (16-30 people)',
  'D: Outside the entrance (30+ people)'
];
const MAX_REASONABLE_WAIT = 90; // minutes; longer entries are kept but flagged

function doGet() {
  return HtmlService.createHtmlOutputFromFile('Index')
    .setTitle('Williams Wait Logger')
    .addMetaTag('viewport', 'width=device-width, initial-scale=1');
}

function setup() {
  const sheet = getSheet_();
  if (sheet.getLastRow() === 0) sheet.appendRow(HEADERS);
}

function getQueueCategories() {
  return QUEUE_CATEGORIES;
}

/**
 * entry = { source: 'crowd'|'manual', joinIso: ISO string, tapIso: ISO string, queue: string }
 * Times come from the phone clock at the moment each button is pressed.
 */
function submitEntry(entry) {
  if (!entry || QUEUE_CATEGORIES.indexOf(entry.queue) === -1) {
    throw new Error('Pick a line-length option.');
  }
  const source = entry.source === 'manual' ? 'manual' : 'crowd';
  const join = new Date(entry.joinIso);
  const tap = new Date(entry.tapIso);
  if (isNaN(join) || isNaN(tap)) throw new Error('Missing a timestamp.');

  const wait = Math.round(((tap - join) / 60000) * 10) / 10;
  let flag = '';
  if (wait < 0) flag = 'negative_wait';
  else if (wait > MAX_REASONABLE_WAIT) flag = 'over_max_wait';

  const tz = Session.getScriptTimeZone();
  const fmt = d => Utilities.formatDate(d, tz, "yyyy-MM-dd'T'HH:mm:ss");
  const lock = LockService.getScriptLock();
  lock.waitLock(10000);
  try {
    getSheet_().appendRow([
      Utilities.getUuid().slice(0, 8), source, fmt(join), fmt(tap),
      entry.queue, wait, fmt(new Date()), flag
    ]);
  } finally {
    lock.releaseLock();
  }
  return { wait: wait, flag: flag };
}

function getSheet_() {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  return ss.getSheetByName(SHEET_NAME) || ss.insertSheet(SHEET_NAME);
}
