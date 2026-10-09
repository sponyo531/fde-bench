/* Interactive views use only the published manuscript data bundled in data.js. */
(() => {
  'use strict';

  const start = () => {
    const data = window.FDE_DATA || { results: [], cases: [], authors: [], affiliations: [] };
    const byId = (id) => document.getElementById(id);
    const element = (tag, className, text) => {
      const node = document.createElement(tag);
      if (className) node.className = className;
      if (text !== undefined) node.textContent = text;
      return node;
    };

    // Arrow keys supplement ordinary Tab/Enter operation for grouped controls.
    const keyboardGroup = (buttons, activate) => {
      buttons.forEach((button, index) => {
        button.addEventListener('keydown', (event) => {
          let next;
          if (event.key === 'ArrowRight' || event.key === 'ArrowDown') next = (index + 1) % buttons.length;
          else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') next = (index - 1 + buttons.length) % buttons.length;
          else if (event.key === 'Home') next = 0;
          else if (event.key === 'End') next = buttons.length - 1;
          else return;
          event.preventDefault();
          buttons[next].focus();
          activate(buttons[next]);
        });
      });
    };

    const head = byId('results-head');
    const body = byId('results-body');
    const resultsMore = byId('results-more');
    const scaffoldFilter = byId('scaffold-filter');
    const modelSearch = byId('model-search');
    const modeButtons = Array.from(document.querySelectorAll('#results-mode button[data-mode]'));
    const resultsState = { mode: 'delivery', metric: 'quality', descending: true, expanded: false };
    const metricGroups = {
      delivery: ['quality', 'dsr', 'pass3', 'all3'],
      clarification: ['quality', 'recall', 'precision', 'askF1']
    };
    const metrics = {
      quality: ['Quality', 'Mean normalized delivery quality'],
      dsr: ['DSR', 'Delivery success rate'],
      pass3: ['pass@3', 'At least one feasible delivery in three runs'],
      all3: ['pass³', 'Feasible deliveries in all three runs'],
      recall: ['Recall', 'Clarification recall'],
      precision: ['Precision', 'Clarification precision'],
      askF1: ['Ask-F1', 'Clarification F1 score']
    };
    const scaffoldNames = {
      opencode: 'OpenCode', openhands: 'OpenHands', codex: 'Codex',
      'deepseek-harness': 'DeepSeek Harness', kimi: 'Kimi', gemini: 'Gemini'
    };

    function renderResults() {
      if (!head || !body) return;
      const columns = metricGroups[resultsState.mode];
      const query = (modelSearch?.value || '').trim().toLowerCase();
      const scaffold = scaffoldFilter?.value || 'all';
      const filtered = data.results.filter((row) => {
        const scaffoldMatches = scaffold === 'all' || (scaffold === 'native'
          ? !['opencode', 'openhands'].includes(row.scaffold) : row.scaffold === scaffold);
        return scaffoldMatches && `${row.model} ${row.scaffold} ${scaffoldNames[row.scaffold] || ''}`.toLowerCase().includes(query);
      }).sort((a, b) => {
        const difference = Number(a[resultsState.metric]) - Number(b[resultsState.metric]);
        return (resultsState.descending ? -difference : difference) || a.model.localeCompare(b.model) || a.scaffold.localeCompare(b.scaffold);
      });

      head.replaceChildren();
      ['Rank', 'Agent / model'].forEach((label) => {
        const th = element('th', '', label);
        th.scope = 'col';
        head.append(th);
      });
      columns.forEach((metric) => {
        const [label, description] = metrics[metric];
        const th = element('th');
        th.scope = 'col';
        const selected = metric === resultsState.metric;
        th.setAttribute('aria-sort', selected ? (resultsState.descending ? 'descending' : 'ascending') : 'none');
        const button = element('button', '', label);
        button.type = 'button';
        button.dataset.metric = metric;
        button.title = description;
        button.setAttribute('aria-label', `Sort by ${description.toLowerCase()}, ${selected && resultsState.descending ? 'ascending' : 'descending'}`);
        const arrow = element('span', 'sort-indicator', selected ? (resultsState.descending ? '↓' : '↑') : '↕');
        arrow.setAttribute('aria-hidden', 'true');
        button.append(arrow);
        button.addEventListener('click', () => {
          resultsState.descending = selected ? !resultsState.descending : true;
          resultsState.metric = metric;
          renderResults();
          head.querySelector(`[data-metric="${metric}"]`)?.focus({ preventScroll: true });
        });
        th.append(button);
        head.append(th);
      });

      body.replaceChildren();
      const shown = resultsState.expanded ? filtered : filtered.slice(0, 6);
      shown.forEach((row, index) => {
        const tr = element('tr');
        tr.append(element('td', 'rank', String(index + 1).padStart(2, '0')));
        const system = element('td', 'system-cell');
        system.append(element('span', 'model-name', row.model));
        system.append(element('span', 'scaffold-name', scaffoldNames[row.scaffold] || row.scaffold));
        tr.append(system);
        columns.forEach((metric) => {
          const value = Number(row[metric]);
          const cell = element('td', metric === 'quality' ? 'cell-quality' : '');
          const formatted = metric === 'quality' ? value.toFixed(4) : `${value.toFixed(2)}%`;
          if (metric === 'quality') {
            cell.append(element('span', 'quality-value', formatted));
            const track = element('span', 'quality-track');
            track.setAttribute('aria-hidden', 'true');
            const fill = element('span', 'quality-fill');
            fill.style.width = `${Math.max(0, Math.min(1, value)) * 100}%`;
            track.append(fill);
            cell.append(track);
          } else cell.textContent = formatted;
          tr.append(cell);
        });
        body.append(tr);
      });
      if (!shown.length) {
        const row = element('tr');
        const cell = element('td', 'empty-state', 'No configurations match these filters. Try another model or scaffold.');
        cell.colSpan = 6;
        row.append(cell);
        body.append(row);
      }
      const count = byId('results-count');
      if (count) count.textContent = `Showing ${shown.length} of ${filtered.length} configurations`;
      if (resultsMore) {
        resultsMore.hidden = filtered.length <= 6;
        resultsMore.textContent = resultsState.expanded ? 'Show top 6 ↑' : `Show all ${filtered.length} configurations ↓`;
        resultsMore.setAttribute('aria-expanded', String(resultsState.expanded));
        resultsMore.setAttribute('aria-controls', 'results-body');
      }
    }
    const changeMode = (button) => {
      resultsState.mode = button.dataset.mode;
      if (!metricGroups[resultsState.mode].includes(resultsState.metric)) {
        resultsState.metric = 'quality';
        resultsState.descending = true;
      }
      modeButtons.forEach((item) => item.setAttribute('aria-pressed', String(item === button)));
      renderResults();
    };
    modeButtons.forEach((button) => button.addEventListener('click', () => changeMode(button)));
    keyboardGroup(modeButtons, changeMode);
    [scaffoldFilter, modelSearch].forEach((control) => control?.addEventListener(control === modelSearch ? 'input' : 'change', () => {
      resultsState.expanded = false;
      renderResults();
    }));
    resultsMore?.addEventListener('click', () => {
      resultsState.expanded = !resultsState.expanded;
      renderResults();
    });
    renderResults();

    const caseGrid = byId('case-grid');
    const caseSearch = byId('case-search');
    const casesMore = byId('cases-more');
    const familyButtons = Array.from(document.querySelectorAll('#case-filters button[data-family]'));
    const caseState = { family: 'all', limit: 6 };
    const featured = ['39', '09', '04', '17', '31', '46'];
    const caseOrder = (item) => featured.includes(item.id) ? featured.indexOf(item.id) : featured.length + Number(item.id);

    function renderCases() {
      if (!caseGrid) return;
      const query = (caseSearch?.value || '').trim().toLowerCase();
      const filtered = data.cases.filter((item) => (caseState.family === 'all' || caseState.family === item.family)
        && `${item.id} ${item.name} ${item.family} ${item.objective} ${item.constraints}`.toLowerCase().includes(query))
        .sort((a, b) => caseOrder(a) - caseOrder(b));
      const shown = filtered.slice(0, caseState.limit);
      caseGrid.replaceChildren();
      shown.forEach((item) => {
        const card = element('article', 'case-card');
        const top = element('div', 'case-top');
        top.append(element('span', 'case-number', `CASE ${item.id}`));
        const badge = element('span', 'family-badge', item.family === 'CO' ? 'OPTIMIZATION' : 'MACHINE LEARNING');
        top.append(badge);
        const heading = element('h3', '', item.name);
        heading.id = `case-heading-${item.id}`;
        card.setAttribute('aria-labelledby', heading.id);
        const objective = element('p', 'case-objective');
        objective.append(element('span', 'case-objective-label', 'Objective'));
        objective.append(document.createTextNode(item.objective));
        const details = element('details');
        details.append(element('summary', '', 'Acceptance checks'));
        details.append(element('p', 'case-checks', item.constraints));
        card.append(top, heading, objective, details);
        caseGrid.append(card);
      });
      if (!shown.length) caseGrid.append(element('p', 'empty-state', 'No cases match your search. Try another keyword or case family.'));
      const count = byId('cases-count');
      if (count) count.textContent = `Showing ${shown.length} of ${filtered.length} cases`;
      if (casesMore) {
        casesMore.hidden = shown.length >= filtered.length;
        casesMore.textContent = `Explore ${Math.min(6, filtered.length - shown.length)} more cases ↓`;
        casesMore.setAttribute('aria-controls', 'case-grid');
      }
    }
    const changeFamily = (button) => {
      caseState.family = button.dataset.family;
      caseState.limit = 6;
      familyButtons.forEach((item) => item.setAttribute('aria-pressed', String(item === button)));
      renderCases();
    };
    familyButtons.forEach((button) => button.addEventListener('click', () => changeFamily(button)));
    keyboardGroup(familyButtons, changeFamily);
    caseSearch?.addEventListener('input', () => { caseState.limit = 6; renderCases(); });
    casesMore?.addEventListener('click', () => { caseState.limit += 6; renderCases(); });
    renderCases();

    const authors = byId('authors');
    if (authors && data.authors.length) {
      authors.replaceChildren();
      data.authors.forEach((author) => {
        const name = element('span', 'author-name', author.name);
        name.append(element('sup', '', `${author.affiliation}${author.mark || ''}`));
        authors.append(name, document.createTextNode(' '));
      });
    }
    const affiliations = byId('affiliations');
    if (affiliations && data.affiliations.length) {
      affiliations.replaceChildren();
      data.affiliations.forEach((affiliation) => {
        const item = element('span', 'affiliation');
        item.append(element('sup', '', affiliation.id), document.createTextNode(affiliation.name));
        affiliations.append(item, document.createTextNode(' '));
      });
    }
    const citation = byId('citation-code');
    if (citation && data.authors.length) {
      citation.textContent = `@misc{li2026fdebench,\n  title = {{FDE-Bench}: Evaluating End-to-End Delivery from Underspecified Real-World Business Requests},\n  author = {${data.authors.map((author) => author.name).join(' and ')}},\n  year = {2026}\n}`;
    }
    const copyStatus = byId('copy-status');
    const legacyCopy = (text) => {
      const active = document.activeElement;
      const field = element('textarea');
      field.value = text;
      field.setAttribute('readonly', '');
      field.style.cssText = 'position:fixed;top:0;left:-9999px;opacity:0;';
      document.body.append(field);
      field.select();
      let copied = false;
      try { copied = document.execCommand('copy'); } catch (_) { /* Manual selection follows. */ }
      field.remove();
      active?.focus({ preventScroll: true });
      return copied;
    };
    byId('copy-citation')?.addEventListener('click', async () => {
      if (!citation) return;
      let copied = false;
      try {
        if (navigator.clipboard?.writeText) {
          await navigator.clipboard.writeText(citation.textContent);
          copied = true;
        }
      } catch (_) { /* Clipboard access can be unavailable for a local preview. */ }
      if (!copied) copied = legacyCopy(citation.textContent);
      if (copyStatus) copyStatus.textContent = copied ? 'Copied!' : 'Selected — press Ctrl+C / ⌘C.';
      if (!copied) {
        const selection = window.getSelection();
        const range = document.createRange();
        range.selectNodeContents(citation);
        selection?.removeAllRanges();
        selection?.addRange(range);
        citation.parentElement?.focus({ preventScroll: true });
      }
    });

    const menu = byId('menu-toggle');
    const nav = byId('site-nav');
    const setMenu = (open) => {
      menu?.setAttribute('aria-expanded', String(open));
      nav?.classList.toggle('is-open', open);
    };
    menu?.addEventListener('click', () => setMenu(menu.getAttribute('aria-expanded') !== 'true'));
    nav?.querySelectorAll('a').forEach((link) => link.addEventListener('click', () => setMenu(false)));
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape' && menu?.getAttribute('aria-expanded') === 'true') {
        setMenu(false);
        menu.focus();
      }
    });

    const storyButtons = Array.from(document.querySelectorAll('#warehouse-tabs button[data-step]'));
    const stories = [
      {
        heading: '“One product type” sounds clear.',
        copy: 'Keep each stack to one product type where possible, while avoiding unnecessary stock moves. But what exactly counts as a mixed stack?',
        note: 'The initial request leaves a business rule unstated.'
      },
      {
        heading: 'The same SKU can still be mixed.',
        copy: 'Different batches of the same product also count as mixed. Sorting by product type alone misses the customer’s actual acceptance rule.',
        note: 'Clarification changes what a valid solution needs to achieve.'
      },
      {
        heading: 'Clarify first. Then deliver.',
        copy: 'Once the mixing rule is known, the agent must optimize stack assignments and stock moves, and deliver an artifact that satisfies the warehouse constraints.',
        note: 'Delivery is judged against the clarified requirements and acceptance checks.'
      }
    ];
    const changeStory = (button) => {
      const step = Number(button.dataset.step);
      const story = stories[step];
      if (!story) return;
      storyButtons.forEach((item) => {
        item.setAttribute('aria-selected', String(item === button));
        item.tabIndex = item === button ? 0 : -1;
      });
      ['heading', 'copy', 'note'].forEach((field) => {
        const target = byId(`warehouse-${field}`);
        if (target) target.textContent = story[field];
      });
      byId('warehouse-panel')?.setAttribute('aria-labelledby', button.id);
      if (byId('warehouse-visual')) byId('warehouse-visual').dataset.step = String(step);
    };
    storyButtons.forEach((button) => button.addEventListener('click', () => changeStory(button)));
    keyboardGroup(storyButtons, changeStory);
  };

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start, { once: true });
  else start();
})();
