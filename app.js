const questionEl = document.querySelector('#question');
const rubricEl = document.querySelector('#rubric');
const answersEl = document.querySelector('#answers');
const countEl = document.querySelector('#answerCount');
const csvFileEl = document.querySelector('#csvFile');
const previewEl = document.querySelector('#answerPreview');
const validationEl = document.querySelector('#answerValidation');
const gradeBtn = document.querySelector('#gradeBtn');
const statusEl = document.querySelector('#status');
const resultsSection = document.querySelector('#results');
let gradedAnswers = [];

function splitAnswers(value) {
  const text = value.trim();
  if (!text) return [];
  return text.split(/\n\s*\n/).map(item => item.trim()).filter(Boolean);
}
function updateAnswerPreview() {
  const count = splitAnswers(answersEl.value).length;
  countEl.textContent = `${count} ${count === 1 ? 'answer' : 'answers'}`;
  const answers = splitAnswers(answersEl.value);
  const errors = [];
  if (answers.some(answer => answer.length > 5000)) errors.push('Each answer must be 5,000 characters or fewer.');
  if (count > 40) errors.push('A session can include up to 40 answers.');
  validationEl.textContent = errors.join(' ');
  previewEl.innerHTML = answers.length ? `<strong>Preview · ${answers.length} ${answers.length === 1 ? 'answer' : 'answers'}</strong><ol>${answers.slice(0, 40).map(answer => `<li>${escapeHtml(answer.length > 240 ? answer.slice(0, 240) + '…' : answer)}</li>`).join('')}</ol>${answers.length > 40 ? '<span>Showing first 40 answers.</span>' : ''}` : '';
}
answersEl.addEventListener('input', updateAnswerPreview);

csvFileEl.addEventListener('change', async () => {
  const file = csvFileEl.files?.[0];
  if (!file) return;
  if (file.size > 2_000_000) { validationEl.textContent = 'CSV file must be smaller than 2 MB.'; csvFileEl.value = ''; return; }
  try {
    const rows = parseCSV(await file.text());
    if (!rows.length) throw new Error('The CSV has no rows.');
    const headers = rows[0].map(value => value.trim().toLowerCase());
    const answerColumn = headers.indexOf('answer');
    if (answerColumn < 0) throw new Error('CSV needs a column named "answer".');
    const values = rows.slice(1).map(row => row[answerColumn] ?? '');
    if (values.some(value => !value.trim())) throw new Error('CSV contains an empty answer row. Fill it in or remove that row.');
    if (!values.length) throw new Error('CSV has a header but no answer rows.');
    answersEl.value = values.join('\n\n');
    validationEl.textContent = `Loaded ${values.length} answers from ${file.name}.`;
    updateAnswerPreview();
  } catch (error) { validationEl.textContent = error.message; }
});

function parseCSV(text) {
  const rows = []; let row = [], cell = '', quoted = false;
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    if (quoted) {
      if (char === '"' && text[i + 1] === '"') { cell += '"'; i++; }
      else if (char === '"') quoted = false;
      else cell += char;
    } else if (char === '"' && cell === '') quoted = true;
    else if (char === ',') { row.push(cell); cell = ''; }
    else if (char === '\n') { row.push(cell.replace(/\r$/, '')); rows.push(row); row = []; cell = ''; }
    else cell += char;
  }
  if (quoted) throw new Error('CSV has an unfinished quoted field.');
  if (cell || row.length) { row.push(cell.replace(/\r$/, '')); rows.push(row); }
  return rows.filter(record => record.some(value => value.trim()));
}

gradeBtn.addEventListener('click', async () => {
  const answers = splitAnswers(answersEl.value);
  updateAnswerPreview();
  if (!questionEl.value.trim() || !rubricEl.value.trim() || !answers.length) {
    statusEl.textContent = 'Please fill in the question, rubric, and at least one student answer.';
    return;
  }
  if (answers.length > 40 || answers.some(answer => answer.length > 5000)) {
    statusEl.textContent = answers.length > 40 ? 'Please grade up to 40 answers per session.' : 'Each answer must be 5,000 characters or fewer. Your question and rubric are still here.';
    return;
  }
  statusEl.textContent = 'Gemini is reviewing the answers against your rubric…';
  gradeBtn.disabled = true;
  gradeBtn.innerHTML = '<span>✳</span> Grading…';
  resultsSection.classList.add('hidden');
  try {
    const response = await fetch('/api/grade', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question: questionEl.value.trim(), rubric: rubricEl.value.trim(), answers})
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Grading request failed.');
    gradedAnswers = data.results;
    renderResults();
    statusEl.textContent = '';
    resultsSection.classList.remove('hidden');
    resultsSection.scrollIntoView({behavior: 'smooth', block: 'start'});
  } catch (error) {
    statusEl.textContent = error.message;
  } finally {
    gradeBtn.disabled = false;
    gradeBtn.innerHTML = '<span>✳</span> Grade answers <b>→</b>';
  }
});

function renderResults() {
  const average = gradedAnswers.reduce((sum, item) => sum + Number(item.score || 0), 0) / gradedAnswers.length;
  const flagged = gradedAnswers.filter(item => item.needs_review).length;
  document.querySelector('#summary').innerHTML = `<div class="summary-chip"><b>${gradedAnswers.length}</b> answers reviewed</div><div class="summary-chip"><b>${average.toFixed(1)}</b> average score</div><div class="summary-chip"><b>${flagged}</b> flagged for review</div>`;
  document.querySelector('#resultList').innerHTML = gradedAnswers.map((item, index) => `
    <article class="result-card">
      <div class="result-number">${String(index + 1).padStart(2, '0')}</div>
      <div><p class="answer-text">${escapeHtml(item.answer)}</p><p class="feedback">${escapeHtml(item.feedback)}</p>
      ${(item.criteria || []).map(c => `<span class="criterion">${escapeHtml(c.name)} · ${Number(c.points_awarded)} / ${Number(c.points_possible)}</span>`).join('')}
      ${item.needs_review ? `<span class="review-flag">Needs your review${item.review_reason ? `: ${escapeHtml(item.review_reason)}` : ''}</span>` : ''}</div>
      <label class="score-box"><input aria-label="Score for answer ${index + 1}" type="number" min="0" max="${Number(item.max_score)}" step="0.5" value="${Number(item.score)}" data-score="${index}"><span>of ${Number(item.max_score)} pts</span></label>
    </article>`).join('');
  document.querySelectorAll('[data-score]').forEach(input => input.addEventListener('change', event => {
    const index = Number(event.target.dataset.score);
    const max = Number(gradedAnswers[index].max_score);
    const score = Math.min(max, Math.max(0, Number(event.target.value) || 0));
    event.target.value = score;
    gradedAnswers[index].score = score;
    renderSummaryOnly();
  }));
}
function renderSummaryOnly() {
  const average = gradedAnswers.reduce((sum, item) => sum + Number(item.score || 0), 0) / gradedAnswers.length;
  const flagged = gradedAnswers.filter(item => item.needs_review).length;
  document.querySelector('#summary').innerHTML = `<div class="summary-chip"><b>${gradedAnswers.length}</b> answers reviewed</div><div class="summary-chip"><b>${average.toFixed(1)}</b> average score</div><div class="summary-chip"><b>${flagged}</b> flagged for review</div>`;
}
function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
}
document.querySelector('#exportBtn').addEventListener('click', () => {
  const rows = [['Answer','Score','Maximum score','Feedback','Needs review'], ...gradedAnswers.map(item => [item.answer, item.score, item.max_score, item.feedback, item.needs_review ? 'Yes' : 'No'])];
  const csv = rows.map(row => row.map(value => `"${String(value ?? '').replaceAll('"','""')}"`).join(',')).join('\r\n');
  const url = URL.createObjectURL(new Blob([csv], {type:'text/csv;charset=utf-8'}));
  const link = document.createElement('a'); link.href = url; link.download = 'smartgrader-results.csv'; link.click(); URL.revokeObjectURL(url);
});
