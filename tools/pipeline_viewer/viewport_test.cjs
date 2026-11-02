// Run: node tools/pipeline_viewer/viewport_test.cjs [runtime-patterns.json]
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync(`${__dirname}/app.js`, 'utf8');
const begin = source.indexOf('function patternVisibleSegments(');
const end = source.indexOf('\nfunction adjustInducedDetail', begin);
const select = vm.runInNewContext(`(${source.slice(begin, end)})`);
const singleton = {start: 0, end: 32, innerCount: 1, innerInterval: 0,
  outerCount: 10000, outerInterval: 64, skipFirst: false};
assert.equal(select(singleton, 65, 95).total, 1);
assert.equal(select(singleton, 33, 63).total, 0);
assert.equal(select({...singleton, skipFirst: true}, 0, 31).total, 0);
const nested = {...singleton, innerCount: 4, innerInterval: 32,
  outerInterval: 256};
assert.equal(select(nested, 257, 287).total, 1);
assert.equal(select(nested, 129, 255).total, 0);
// Wide SXM jobs last 70 cycles but can start every 32. A viewport after
// the last issue must still show the in-flight output tail.
const transpose = {start: 100, end: 170, innerCount: 4, innerInterval: 32,
  outerCount: 1, outerInterval: 0, skipFirst: false};
assert.equal(select(transpose, 200, 201).total, 3);
assert.equal(select(transpose, 265, 266).total, 1);
assert.equal(select(transpose, 267, 268).total, 0);
if (process.argv[2]) {
  const patterns = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
  // Zoomed projection must stay fully occupied without exceeding a 600-event
  // row budget. Offscreen repeats must not force the renderer to subsample.
  for (const start of [38372, 45000, 60000, 70000]) {
    const finish = Math.min(start + 640, 71140);
    const intervals = [];
    let total = 0;
    for (const p of patterns) {
      const info = select(p, start, finish);
      total += info.total;
      for (const s of info.segments)
        for (let i = s.first; i < s.first + s.count; ++i) {
          const t = p.start + s.outer * p.outerInterval + i * p.innerInterval;
          intervals.push([t, t + p.end - p.start]);
        }
    }
    assert(total < 600, `false LOD at ${start}: ${total}`);
    intervals.sort((a,b) => a[0]-b[0]);
    let covered = start;
    for (const [a,b] of intervals) {
      assert(a <= covered, `false bubble at ${covered}`);
      covered = Math.max(covered,b);
    }
    assert(covered >= finish);
  }
}
console.log('Viewport clipping and projection continuity: passed');
