console.log('app.js loaded');
const API_BASE = '/api';
// LOK_NO kept for any remaining per-lok references; MP list now loads from combined endpoint
const LOK_NO = 18;

// Feature flag: clean card text display
// When true: show only first C chunks per card, strip repeated headers (header shown once at top)
// When false: show all chunks as-is with '---' separators (original behaviour)
const CLEAN_CARD_TEXT = true;

const elements = {
    form: document.getElementById('search-form'),
    queryInput: document.getElementById('query-input'),
    submitBtn: document.getElementById('submit-btn'),
    aiMode: document.getElementById('ai-mode'),
    topkInput: document.getElementById('topk-input'),
    topnInput: document.getElementById('topn-input'),
    cpqInput: document.getElementById('cpq-input'),
    results: document.getElementById('results'),
    answerSection: document.getElementById('answer-section'),
    answerContent: document.getElementById('answer-content'),
    evidenceTitle: document.getElementById('evidence-title'),
    evidenceCount: document.getElementById('evidence-count'),
    evidenceCards: document.getElementById('evidence-cards'),
    error: document.getElementById('error'),
    errorMessage: document.getElementById('error-message'),
    modal: document.getElementById('modal'),
    modalClose: document.getElementById('modal-close'),
    modalTitle: document.getElementById('modal-title'),
    modalChips: document.getElementById('modal-chips'),
    modalBody: document.getElementById('modal-body'),
    modalFooter: document.getElementById('modal-footer'),
    mpDropdown: document.getElementById('mp-dropdown'),
    activeFilters: document.getElementById('active-filters'),
    topqInput: document.getElementById('topq-input'),
    resultsLayout: document.getElementById('results-layout'),
    statsPanel: document.getElementById('stats-panel'),
    statsContent: document.getElementById('stats-content'),
};

let currentEvidence = [];

// Cache for lazy-fetched absolute leading chunk text: key = "lok-sess-qno-cN"
const _questionTextCache = new Map();

// --- MP autocomplete state ---

let mpList = [];          // full list of MP names from API
let dropdownIndex = -1;   // keyboard navigation index in dropdown
let mpFilter = null;      // currently selected MP name for Qdrant filtering

async function loadMpList() {
    try {
        const res = await fetch(`${API_BASE}/members`);
        if (res.ok) {
            mpList = await res.json();
            console.log(`Loaded ${mpList.length} MP names for autocomplete`);
        } else {
            console.warn('Failed to load MP list:', res.status);
        }
    } catch (err) {
        console.warn('Failed to load MP list:', err);
    }
}

function getMpTrigger() {
    const val = elements.queryInput.value;
    const cursorPos = elements.queryInput.selectionStart;
    const textBeforeCursor = val.slice(0, cursorPos);

    const match = textBeforeCursor.match(/(?:^|\s)mp:(\S*)$/i);
    if (!match) return null;

    return {
        search: match[1].toLowerCase(),
        tokenStart: match.index + (match[0].startsWith(' ') ? 1 : 0),
        tokenEnd: cursorPos,
    };
}

function fuzzyMatch(name, search) {
    return name.toLowerCase().includes(search);
}

function highlightMatch(name, search) {
    if (!search) return name;
    const idx = name.toLowerCase().indexOf(search);
    if (idx === -1) return name;
    return name.slice(0, idx) +
        `<span class="mp-match">${name.slice(idx, idx + search.length)}</span>` +
        name.slice(idx + search.length);
}

function showDropdown(trigger) {
    if (!trigger || mpList.length === 0) {
        hideDropdown();
        return;
    }

    const matches = mpList
        .filter(name => fuzzyMatch(name, trigger.search))
        .slice(0, 8);

    if (matches.length === 0) {
        elements.mpDropdown.innerHTML = '<div class="mp-dropdown-item" style="color: var(--color-text-muted)">No matching MPs</div>';
        elements.mpDropdown.classList.add('visible');
        dropdownIndex = -1;
        return;
    }

    elements.mpDropdown.innerHTML = matches.map((name, i) =>
        `<div class="mp-dropdown-item${i === dropdownIndex ? ' active' : ''}" data-index="${i}" data-name="${name}">${highlightMatch(name, trigger.search)}</div>`
    ).join('') + '<div class="mp-dropdown-hint">Tab or Enter to select</div>';

    elements.mpDropdown.classList.add('visible');

    // Click handler on items
    elements.mpDropdown.querySelectorAll('.mp-dropdown-item[data-name]').forEach(el => {
        el.addEventListener('mousedown', (e) => {
            e.preventDefault(); // prevent blur
            selectMp(el.dataset.name, trigger);
        });
    });
}

function hideDropdown() {
    elements.mpDropdown.classList.remove('visible');
    elements.mpDropdown.innerHTML = '';
    dropdownIndex = -1;
}

function selectMp(name, trigger) {
    // Replace "mp:..." token with the full MP name inline
    const val = elements.queryInput.value;
    const before = val.slice(0, trigger.tokenStart);
    const after = val.slice(trigger.tokenEnd);
    elements.queryInput.value = before + name + after;

    // Place cursor right after the inserted name
    const cursorPos = before.length + name.length;
    elements.queryInput.setSelectionRange(cursorPos, cursorPos);

    // Set as active filter for Qdrant metadata filtering
    mpFilter = name;
    renderActiveFilters();
    console.log('MP filter set:', name);

    hideDropdown();
    elements.queryInput.focus();
}

function renderActiveFilters() {
    if (!mpFilter) {
        elements.activeFilters.classList.remove('visible');
        elements.activeFilters.innerHTML = '';
        return;
    }

    elements.activeFilters.classList.add('visible');
    elements.activeFilters.innerHTML = `
        <span class="filter-label">Filtering by:</span>
        <span class="filter-tag">
            MP: ${mpFilter}
            <button type="button" class="filter-tag-remove" title="Remove filter">\u00d7</button>
        </span>
    `;

    elements.activeFilters.querySelector('.filter-tag-remove').addEventListener('click', () => {
        mpFilter = null;
        renderActiveFilters();
        console.log('MP filter cleared');
    });
}

function clearMpFilter() {
    mpFilter = null;
    renderActiveFilters();
}

function getVisibleDropdownItems() {
    return elements.mpDropdown.querySelectorAll('.mp-dropdown-item[data-name]');
}

function handleDropdownKeyboard(e) {
    const isVisible = elements.mpDropdown.classList.contains('visible');
    if (!isVisible) return false;

    const items = getVisibleDropdownItems();
    if (items.length === 0) return false;

    if (e.key === 'ArrowDown') {
        e.preventDefault();
        dropdownIndex = Math.min(dropdownIndex + 1, items.length - 1);
        updateDropdownActive(items);
        return true;
    }

    if (e.key === 'ArrowUp') {
        e.preventDefault();
        dropdownIndex = Math.max(dropdownIndex - 1, 0);
        updateDropdownActive(items);
        return true;
    }

    if (e.key === 'Tab' || e.key === 'Enter') {
        const trigger = getMpTrigger();
        if (!trigger) return false;

        e.preventDefault();

        if (dropdownIndex >= 0 && dropdownIndex < items.length) {
            selectMp(items[dropdownIndex].dataset.name, trigger);
        } else if (items.length > 0) {
            selectMp(items[0].dataset.name, trigger);
        }
        return true;
    }

    if (e.key === 'Escape') {
        hideDropdown();
        return true;
    }

    return false;
}

function updateDropdownActive(items) {
    items.forEach((el, i) => {
        el.classList.toggle('active', i === dropdownIndex);
    });
}

// --- Shared helpers ---

function setLoading(loading) {
    elements.submitBtn.disabled = loading;
    if (loading) {
        elements.submitBtn.classList.add('loading');
    } else {
        elements.submitBtn.classList.remove('loading');
    }
}

function showError(message) {
    elements.results.classList.remove('visible');
    elements.error.classList.add('visible');
    elements.errorMessage.textContent = message;
}

function hideError() {
    elements.error.classList.remove('visible');
}

function formatCitations(text) {
    return text.replace(/\[Q(\d+)\]/g, (match, num) => {
        return `<span class="citation" data-index="${num}">${match}</span>`;
    });
}

function formatAskedBy(askedBy) {
    if (!askedBy) return null;

    let names = [];
    if (typeof askedBy === 'string') {
        if (askedBy.startsWith('[')) {
            try {
                names = JSON.parse(askedBy.replace(/'/g, '"'));
            } catch {
                names = askedBy.split(',').map(s => s.trim());
            }
        } else {
            names = askedBy.split(',').map(s => s.trim());
        }
    } else if (Array.isArray(askedBy)) {
        names = askedBy;
    }

    names = names.filter(n => n && n.trim());
    if (names.length === 0) return null;

    if (names.length === 1) {
        return names[0];
    }
    return `${names[0]} + ${names.length - 1} more`;
}

function renderAnswer(answer) {
    elements.answerSection.classList.add('visible');
    const html = marked.parse(answer);
    elements.answerContent.innerHTML = formatCitations(html);

    elements.answerContent.querySelectorAll('.citation').forEach(el => {
        el.addEventListener('click', () => {
            const index = parseInt(el.dataset.index, 10);
            scrollToEvidence(index);
        });
    });
}

// --- Synthesizing indicator ---

let _synthTimer = null;

const SYNTH_STAGES = [
    'Collecting questions',
    'Analysing sources',
    'Collating answers',
    'Drafting response',
];

function showSynthesizingIndicator() {
    elements.results.classList.add('visible');
    elements.answerSection.classList.add('visible');
    elements.answerContent.innerHTML =
        '<div class="synth-indicator"><span id="synth-stage-text"></span></div>';

    let stageIdx = 0;
    let dots = 1;
    const textEl = document.getElementById('synth-stage-text');

    const render = () => {
        textEl.textContent = SYNTH_STAGES[stageIdx] + '.'.repeat(dots);
    };
    render();

    // Advance stage every 2 seconds (stop at last stage)
    const stageTimer = setInterval(() => {
        if (stageIdx < SYNTH_STAGES.length - 1) stageIdx++;
        render();
    }, 2000);

    // Animate dots 1 → 2 → 3 → 1 every 500ms
    const dotTimer = setInterval(() => {
        dots = dots >= 3 ? 1 : dots + 1;
        render();
    }, 500);

    _synthTimer = { stageTimer, dotTimer };
}

function hideSynthesizingIndicator() {
    if (_synthTimer) {
        clearInterval(_synthTimer.stageTimer);
        clearInterval(_synthTimer.dotTimer);
        _synthTimer = null;
    }
    const el = document.querySelector('.synth-indicator');
    if (el) el.remove();
}

function showStatsPanel() {
    document.querySelector('.container').classList.add('has-stats');
    elements.resultsLayout.classList.add('has-stats');
    elements.statsPanel.classList.add('visible');
}

function hideStatsPanel() {
    document.querySelector('.container').classList.remove('has-stats');
    elements.resultsLayout.classList.remove('has-stats');
    elements.statsPanel.classList.remove('visible');
    elements.statsContent.innerHTML = '';
}

function renderStatsProgressive(mpStats) {
    if (!mpStats) {
        hideStatsPanel();
        return;
    }

    showStatsPanel();
    elements.statsContent.innerHTML = '';

    // Phase 1: Header (MP name + big number + lok/type chips) — immediate
    const lokHTML = Object.entries(mpStats.by_lok)
        .map(([k, v]) => `<span class="chip">Lok Sabha ${k}: ${v}</span>`)
        .join('');
    const typeHTML = Object.entries(mpStats.by_type)
        .map(([k, v]) => `<span class="chip secondary">${k}: ${v}</span>`)
        .join('');

    const headerEl = document.createElement('div');
    headerEl.className = 'stats-header stats-phase';
    headerEl.innerHTML = `
        <div class="stats-total">
            <span class="stats-mp-name">${mpStats.mp_name}</span>
            <span class="stats-big-number">${mpStats.total_questions}</span>
            <span class="stats-label">total questions</span>
        </div>
        <div class="stats-breakdown">
            <div class="stats-sessions">${lokHTML}</div>
            <div class="stats-types">${typeHTML}</div>
        </div>
    `;
    elements.statsContent.appendChild(headerEl);

    // Phase 2: Ministry bar chart — staggered
    const maxMinCount = mpStats.top_ministries.length > 0
        ? mpStats.top_ministries[0].count : 1;
    const ministryHTML = mpStats.top_ministries.slice(0, 10).map(m =>
        `<div class="stats-bar">
            <span class="stats-bar-label" title="${m.ministry}">${m.ministry}</span>
            <div class="stats-bar-track">
                <div class="stats-bar-fill" data-width="${(m.count / maxMinCount * 100).toFixed(0)}%"></div>
            </div>
            <span class="stats-bar-count">${m.count}</span>
        </div>`
    ).join('');

    const ministriesEl = document.createElement('div');
    ministriesEl.className = 'stats-ministries stats-phase';
    ministriesEl.innerHTML = `<h4>Top Ministries</h4>${ministryHTML}`;
    elements.statsContent.appendChild(ministriesEl);

    // Animate bar widths after DOM paint
    requestAnimationFrame(() => {
        ministriesEl.querySelectorAll('.stats-bar-fill').forEach(bar => {
            bar.style.width = bar.dataset.width;
        });
    });

    // Phase 3: Recent questions — staggered
    if (mpStats.recent_questions.length > 0) {
        const recentHTML = mpStats.recent_questions.map(q =>
            `<li class="stats-question">
                <span class="stats-q-subject">${q.subject}</span>
                <span class="stats-q-meta">Lok ${q.lok_no}, Session ${q.session_no}, Q${q.ques_no} &middot; ${q.ministry} &middot; ${q.date || 'N/A'}</span>
            </li>`
        ).join('');

        const recentEl = document.createElement('div');
        recentEl.className = 'stats-recent stats-phase';
        recentEl.innerHTML = `<h4>Recent Questions</h4><ol>${recentHTML}</ol>`;
        elements.statsContent.appendChild(recentEl);
    }
}

function scrollToEvidence(index) {
    const card = document.querySelector(`.evidence-card[data-group-index="${index}"]`);
    if (card) {
        document.querySelectorAll('.evidence-card').forEach(c => c.classList.remove('highlighted'));
        card.classList.add('highlighted');
        card.scrollIntoView({ behavior: 'smooth', block: 'center' });
        setTimeout(() => card.classList.remove('highlighted'), 2000);
    }
}

// --- Client-side grouping for search mode (flat results) ---

function groupByQuestionFlat(evidence) {
    const groups = new Map();

    for (const item of evidence) {
        const key = `${item.lok_no}-${item.session_no}-${item.type || ''}-${item.ques_no}`;
        if (!groups.has(key)) {
            groups.set(key, {
                ...item,
                indices: [item.index],
                chunks: [item],
                best_score: item.score,
            });
        } else {
            const group = groups.get(key);
            group.indices.push(item.index);
            group.chunks.push(item);
            if (item.score > group.best_score) {
                group.best_score = item.score;
                group.subject = item.subject || group.subject;
                group.text_preview = item.text_preview || group.text_preview;
            }
        }
    }

    return Array.from(groups.values()).sort((a, b) => b.best_score - a.best_score);
}

// --- Shared card helpers ---

function buildChips(group) {
    const chips = [];

    if (group.lok_no) {
        chips.push({ label: `Lok Sabha ${group.lok_no}`, type: 'default' });
    }

    if (group.session_no) {
        chips.push({ label: `Session ${group.session_no}`, type: 'default' });
    }

    if (group.ques_no) {
        chips.push({ label: `Q${group.ques_no}`, type: 'default' });
    }

    return chips;
}

function buildMetaText(group) {
    const parts = [];

    if (group.ministry) {
        parts.push(`Ministry: ${group.ministry}`);
    }

    const askedBy = formatAskedBy(group.asked_by);
    if (askedBy) {
        parts.push(`Asked by: ${askedBy}`);
    }

    return parts;
}

function buildSecondaryChips(group) {
    const chips = [];

    chips.push({ label: `${(group.best_score * 100).toFixed(0)}% match`, type: 'secondary' });

    if (group.chunks.length > 1) {
        chips.push({ label: `${group.chunks.length} parts`, type: 'secondary' });
    }

    // Show how many trailing chunks were trimmed
    const total = group.total_chunks_available || group.chunks.length;
    const trimmed = total - group.chunks.length;
    if (trimmed > 0) {
        chips.push({ label: `+${trimmed} more in PDF`, type: 'secondary' });
    }

    return chips;
}

function renderChipsHTML(chips) {
    return chips.map(c => {
        const cls = c.type === 'primary' ? 'chip primary' : c.type === 'secondary' ? 'chip secondary' : 'chip';
        return `<span class="${cls}">${c.label}</span>`;
    }).join('');
}

function getCleanCardText(chunks, maxC) {
    // Sort by chunk_index, take first maxC chunks
    const sorted = chunks
        .slice()
        .sort((a, b) => (a.chunk_index || 0) - (b.chunk_index || 0))
        .slice(0, maxC);

    if (sorted.length === 0) return '';

    // First chunk: keep header + body intact
    // Subsequent chunks: strip the header (everything up to and including the first blank line)
    const parts = sorted.map((chunk, i) => {
        if (i === 0) return chunk.text;
        const sep = chunk.text.indexOf('\n\n');
        return sep !== -1 ? chunk.text.slice(sep + 2) : chunk.text;
    });

    return parts.join('\n\n');
}

function getCombinedText(chunks) {
    if (CLEAN_CARD_TEXT) {
        const maxC = parseInt(elements.cpqInput?.value, 10) || 1;
        return getCleanCardText(chunks, maxC);
    }
    // Original behaviour: all chunks with repeated headers and '---' separators
    return chunks
        .slice()
        .sort((a, b) => (a.chunk_index || 0) - (b.chunk_index || 0))
        .map(c => c.text)
        .join('\n\n---\n\n');
}

// Fetch the absolute leading C chunks for a question from the backend.
// Uses a module-level cache so repeated opens/expands don't re-fetch.
async function fetchQuestionText(group, c) {
    const key = group.question_id
        ? `${group.question_id}-c${c}`
        : `${group.lok_no}-${group.session_no}-${group.type || ''}-${group.ques_no}-c${c}`;
    if (_questionTextCache.has(key)) {
        return _questionTextCache.get(key);
    }
    try {
        const payload = { c };
        if (group.question_id) {
            payload.question_id = group.question_id;
        } else {
            payload.lok_no = group.lok_no;
            payload.session_no = group.session_no;
            payload.ques_no = group.ques_no;
            if (group.type) payload.type = group.type;
        }

        const res = await fetch(`${API_BASE}/question-text`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        if (!res.ok) return null;
        const data = await res.json();
        _questionTextCache.set(key, data.text);
        return data.text;
    } catch {
        return null;
    }
}

// --- Render evidence cards ---

function renderEvidenceCards(data, isGrouped) {
    currentEvidence = data;
    elements.evidenceCards.innerHTML = '';

    if (isGrouped) {
        renderEvidenceCardsGrouped(data);
    } else {
        renderEvidenceCardsFlat(data);
    }
}

function renderEvidenceCardsGrouped(evidenceGroups) {
    const totalChunks = evidenceGroups.reduce((sum, g) => sum + g.chunks.length, 0);
    elements.evidenceCount.textContent = `(${evidenceGroups.length} questions from ${totalChunks} chunks)`;

    evidenceGroups.forEach((group) => {
        const card = document.createElement('div');
        card.className = 'evidence-card';
        card.dataset.groupIndex = group.group_index;

        const chips = buildChips(group);
        const metaText = buildMetaText(group);
        const secondaryChips = buildSecondaryChips(group);

        // In CLEAN_CARD_TEXT mode, card-text is lazy-loaded on first expand.
        // In original mode, pre-render now so expand is instant.
        const preRendered = CLEAN_CARD_TEXT ? '' : marked.parse(getCombinedText(group.chunks));

        card.innerHTML = `
            <div class="card-main">
                <div class="card-header">
                    <div class="card-title">${group.subject || 'Parliamentary Question'}</div>
                    <div class="card-chips">${renderChipsHTML(chips)}</div>
                </div>
                <div class="card-meta">${metaText.map(t => `<span>${t}</span>`).join('')}</div>
                <div class="card-footer">
                    <div class="card-secondary">${renderChipsHTML(secondaryChips)}</div>
                    ${group.pdf_url ? `<a href="${group.pdf_url}" target="_blank" class="card-link" onclick="event.stopPropagation()">View PDF</a>` : ''}
                </div>
            </div>
            <div class="card-expand">
                <button class="card-expand-toggle" data-card="${group.group_index}">
                    <span>Show full text</span>
                    <span class="arrow">&#9662;</span>
                </button>
                <div class="card-text" id="card-text-g${group.group_index}">${preRendered}</div>
            </div>
        `;

        card.querySelector('.card-main').addEventListener('click', () => openModal(group));

        const toggleBtn = card.querySelector('.card-expand-toggle');
        const textEl = card.querySelector('.card-text');

        toggleBtn.addEventListener('click', async (e) => {
            e.stopPropagation();
            const isExpanded = textEl.classList.contains('visible');

            if (!isExpanded && CLEAN_CARD_TEXT && !textEl.dataset.loaded) {
                // First expand in clean mode: lazy-fetch absolute leading C chunks
                const maxC = parseInt(elements.cpqInput?.value, 10) || 1;
                textEl.innerHTML = '<div class="card-text-loading">Loading\u2026</div>';
                textEl.classList.add('visible');
                toggleBtn.classList.add('expanded');
                toggleBtn.querySelector('span:first-child').textContent = 'Hide full text';

                if (!textEl.dataset.loading) {
                    textEl.dataset.loading = '1';
                    const text = await fetchQuestionText(group, maxC);
                    textEl.innerHTML = text
                        ? marked.parse(text)
                        : '<em style="color:var(--color-text-muted)">No content found for this question.</em>';
                    textEl.dataset.loaded = '1';
                    delete textEl.dataset.loading;
                }
            } else {
                // Toggle: collapsing, or already loaded — just flip visibility
                textEl.classList.toggle('visible');
                toggleBtn.classList.toggle('expanded');
                toggleBtn.querySelector('span:first-child').textContent =
                    textEl.classList.contains('visible') ? 'Hide full text' : 'Show full text';
            }
        });

        elements.evidenceCards.appendChild(card);
    });
}

function renderEvidenceCardsFlat(evidence) {
    const grouped = groupByQuestionFlat(evidence);
    elements.evidenceCount.textContent = `(${grouped.length} questions from ${evidence.length} chunks)`;

    grouped.forEach((group, idx) => {
        const card = document.createElement('div');
        card.className = 'evidence-card';
        card.dataset.indices = JSON.stringify(group.indices);

        const chips = buildChips(group);
        const metaText = buildMetaText(group);
        const secondaryChips = buildSecondaryChips(group);

        const preRendered = CLEAN_CARD_TEXT ? '' : marked.parse(getCombinedText(group.chunks));

        card.innerHTML = `
            <div class="card-main">
                <div class="card-header">
                    <div class="card-title">${group.subject || 'Parliamentary Question'}</div>
                    <div class="card-chips">${renderChipsHTML(chips)}</div>
                </div>
                <div class="card-meta">${metaText.map(t => `<span>${t}</span>`).join('')}</div>
                <div class="card-footer">
                    <div class="card-secondary">${renderChipsHTML(secondaryChips)}</div>
                    ${group.pdf_url ? `<a href="${group.pdf_url}" target="_blank" class="card-link" onclick="event.stopPropagation()">View PDF</a>` : ''}
                </div>
            </div>
            <div class="card-expand">
                <button class="card-expand-toggle" data-card="${idx}">
                    <span>Show full text</span>
                    <span class="arrow">&#9662;</span>
                </button>
                <div class="card-text" id="card-text-${idx}">${preRendered}</div>
            </div>
        `;

        card.querySelector('.card-main').addEventListener('click', () => openModal(group));

        const toggleBtn = card.querySelector('.card-expand-toggle');
        const textEl = card.querySelector('.card-text');

        toggleBtn.addEventListener('click', async (e) => {
            e.stopPropagation();
            const isExpanded = textEl.classList.contains('visible');

            if (!isExpanded && CLEAN_CARD_TEXT && !textEl.dataset.loaded) {
                const maxC = parseInt(elements.cpqInput?.value, 10) || 1;
                textEl.innerHTML = '<div class="card-text-loading">Loading\u2026</div>';
                textEl.classList.add('visible');
                toggleBtn.classList.add('expanded');
                toggleBtn.querySelector('span:first-child').textContent = 'Hide full text';

                if (!textEl.dataset.loading) {
                    textEl.dataset.loading = '1';
                    const text = await fetchQuestionText(group, maxC);
                    textEl.innerHTML = text
                        ? marked.parse(text)
                        : '<em style="color:var(--color-text-muted)">No content found for this question.</em>';
                    textEl.dataset.loaded = '1';
                    delete textEl.dataset.loading;
                }
            } else {
                textEl.classList.toggle('visible');
                toggleBtn.classList.toggle('expanded');
                toggleBtn.querySelector('span:first-child').textContent =
                    textEl.classList.contains('visible') ? 'Hide full text' : 'Show full text';
            }
        });

        elements.evidenceCards.appendChild(card);
    });
}

// --- Modal ---

async function openModal(group) {
    elements.modalTitle.textContent = group.subject || 'Parliamentary Question';

    const chips = buildChips(group);
    const metaText = buildMetaText(group);
    const secondaryChips = buildSecondaryChips(group);

    let chipsHTML = renderChipsHTML(chips);
    if (metaText.length > 0) {
        chipsHTML += `<span class="chip-divider"></span>`;
        chipsHTML += metaText.map(t => `<span class="chip">${t}</span>`).join('');
    }
    chipsHTML += `<span class="chip-divider"></span>`;
    chipsHTML += renderChipsHTML(secondaryChips);

    elements.modalChips.innerHTML = chipsHTML;

    if (group.pdf_url) {
        elements.modalFooter.innerHTML = `<a href="${group.pdf_url}" target="_blank">View Original PDF &rarr;</a>`;
    } else {
        elements.modalFooter.innerHTML = '';
    }

    // Open modal immediately — don't wait for content
    elements.modal.classList.add('open');
    document.body.style.overflow = 'hidden';

    if (CLEAN_CARD_TEXT) {
        // Lazy-fetch absolute leading C chunks from Qdrant (cache shared with card expand)
        const maxC = parseInt(elements.cpqInput?.value, 10) || 1;
        elements.modalBody.innerHTML = '<div class="card-text-loading">Loading\u2026</div>';
        const text = await fetchQuestionText(group, maxC);
        elements.modalBody.innerHTML = text
            ? marked.parse(text)
            : '<em style="color:var(--color-text-muted)">No content found for this question.</em>';
    } else {
        const combinedText = getCombinedText(group.chunks);
        elements.modalBody.innerHTML = marked.parse(combinedText);
    }
}

function closeModal() {
    elements.modal.classList.remove('open');
    document.body.style.overflow = '';
}

// --- Submit handler ---

async function handleSubmit(e) {
    e.preventDefault();
    hideError();

    const query = elements.queryInput.value.trim();
    if (!query) return;

    const useAI = elements.aiMode.checked;
    const topK = parseInt(elements.topkInput.value, 10) || 30;
    const topN = parseInt(elements.topnInput.value, 10) || 10;
    const cpq = parseInt(elements.cpqInput.value, 10) || 2;
    const topQ = parseInt(elements.topqInput.value, 10) || 10;

    const payload = {
        query: query,
        top_k: topK,
    };

    // Only send N, C, and Q for synthesize mode (search returns flat results)
    if (useAI) {
        payload.top_n = topN;
        payload.chunks_per_question = cpq;
        payload.top_q = topQ;
    }

    // MP name metadata filter
    if (mpFilter) {
        payload.mp_names = [mpFilter];
    }

    setLoading(true);
    elements.results.classList.remove('visible');
    elements.answerSection.classList.remove('visible');
    hideStatsPanel();

    // Show phased thinking indicator immediately for AI mode
    if (useAI) {
        showSynthesizingIndicator();
    }

    // Fire stats fetch in parallel if MP filter is active (regardless of AI mode)
    const shouldFetchStats = !!mpFilter;
    let statsPromise = null;
    if (shouldFetchStats) {
        statsPromise = fetch(`${API_BASE}/mp-stats`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mp_name: mpFilter, top_q: topQ }),
        })
        .then(res => res.ok ? res.json() : null)
        .catch(() => null);

        // Show stats panel as soon as they arrive (don't wait for synthesize)
        statsPromise.then(statsData => {
            if (statsData) {
                // Make results area visible so the grid layout activates
                elements.results.classList.add('visible');
                renderStatsProgressive(statsData);
            }
        });
    }

    try {
        const endpoint = useAI ? '/synthesize' : '/search';
        const response = await fetch(`${API_BASE}${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `Request failed (${response.status})`);
        }

        const data = await response.json();

        elements.results.classList.add('visible');

        if (useAI) {
            hideSynthesizingIndicator();
            renderAnswer(data.answer);
            // If stats weren't fetched in parallel (edge case), use inline data
            if (!shouldFetchStats && data.mp_stats) {
                renderStatsProgressive(data.mp_stats);
            } else if (!shouldFetchStats) {
                hideStatsPanel();
            }
            elements.evidenceTitle.textContent = 'Sources';
            renderEvidenceCards(data.evidence_groups, true);
        } else {
            elements.answerSection.classList.remove('visible');
            elements.evidenceTitle.textContent = 'Results';
            renderEvidenceCards(data.results, false);
        }
    } catch (err) {
        hideSynthesizingIndicator();
        showError(err.message);
    } finally {
        setLoading(false);
    }
}

// --- Event listeners ---

elements.form.addEventListener('submit', (e) => {
    // If dropdown is open and user presses Enter, it should select, not submit
    const trigger = getMpTrigger();
    if (trigger && elements.mpDropdown.classList.contains('visible')) {
        e.preventDefault();
        const items = getVisibleDropdownItems();
        if (items.length > 0) {
            const idx = dropdownIndex >= 0 ? dropdownIndex : 0;
            selectMp(items[idx].dataset.name, trigger);
        }
        return;
    }
    handleSubmit(e);
});

elements.queryInput.addEventListener('input', () => {
    const trigger = getMpTrigger();
    if (trigger) {
        console.log('MP trigger detected:', trigger.search, `(${mpList.length} MPs loaded)`);
        dropdownIndex = -1;
        showDropdown(trigger);
    } else {
        hideDropdown();
    }
});

elements.queryInput.addEventListener('keydown', (e) => {
    if (handleDropdownKeyboard(e)) return;
});

elements.queryInput.addEventListener('blur', () => {
    setTimeout(hideDropdown, 150);
});

elements.modalClose.addEventListener('click', closeModal);
elements.modal.querySelector('.modal-backdrop').addEventListener('click', closeModal);

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && elements.modal.classList.contains('open')) {
        closeModal();
    }
});

// --- Init ---

elements.queryInput.focus();
loadMpList();
