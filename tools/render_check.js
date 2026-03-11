const fs = require('fs');

function readPayload() {
  const raw = fs.readFileSync(0, 'utf8').trim();
  if (!raw) {
    return { backend: 'mathjax', expressions: [] };
  }
  return JSON.parse(raw);
}

function checkWithKatex(expressions) {
  const katex = require('katex');
  return expressions.map((expr) => {
    try {
      katex.renderToString(expr, {
        throwOnError: true,
        strict: 'error',
        displayMode: true,
      });
      return { ok: true };
    } catch (error) {
      return { ok: false, error: error.message };
    }
  });
}

function checkWithMathJax(expressions) {
  const { mathjax } = require('mathjax-full/js/mathjax.js');
  const { TeX } = require('mathjax-full/js/input/tex.js');
  const { SVG } = require('mathjax-full/js/output/svg.js');
  const { liteAdaptor } = require('mathjax-full/js/adaptors/liteAdaptor.js');
  const { RegisterHTMLHandler } = require('mathjax-full/js/handlers/html.js');
  const { AllPackages } = require('mathjax-full/js/input/tex/AllPackages.js');

  const adaptor = liteAdaptor();
  RegisterHTMLHandler(adaptor);
  const tex = new TeX({
    packages: AllPackages,
    formatError: (_jax, error) => {
      throw error;
    },
  });
  const svg = new SVG({ fontCache: 'none' });
  const html = mathjax.document('', { InputJax: tex, OutputJax: svg });

  return expressions.map((expr) => {
    try {
      html.convert(expr, { display: true });
      return { ok: true };
    } catch (error) {
      return { ok: false, error: error.message };
    }
  });
}

function main() {
  const payload = readPayload();
  const backend = payload.backend || 'mathjax';
  const expressions = payload.expressions || [];

  let results;
  if (backend === 'katex') {
    results = checkWithKatex(expressions);
  } else if (backend === 'mathjax') {
    results = checkWithMathJax(expressions);
  } else {
    throw new Error(`Unsupported render backend: ${backend}`);
  }

  process.stdout.write(JSON.stringify(results));
}

main();
