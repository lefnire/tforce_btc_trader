const puppeteer = require('puppeteer');
const path = require('path');

(async() => {
const browser = await puppeteer.launch();
const page = await browser.newPage();
await page.goto('file:///' + path.resolve(__dirname, 'embed.html'), {waitUntil: 'networkidle'});
setInterval(() => {
  // await page.screenshot()
  page.screenshot({path: 'images/' + +new Date + '.png'});
}, 5000);
//browser.close();
})();