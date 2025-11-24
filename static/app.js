
// Referencias comunes
const startBtn = document.getElementById("startBtn");
const rowsInput = document.getElementById("rowsInput");
const forceChk = document.getElementById("forceChk");
const btnCsv = document.getElementById("btnCsv");
const btnJson = document.getElementById("btnJson");
const stepsEl = document.getElementById("steps");
const resultEl = document.getElementById("result");
const errorEl = document.getElementById("error");
const barEl = document.getElementById("bar");
// Resumen
const rowsEl = document.getElementById("rows");
const colsEl = document.getElementById("cols");
const xcolsEl = document.getElementById("xcols");
const ycolsEl = document.getElementById("ycols");
const noteEl = document.getElementById("note");
// Métricas logística
const logAccEl = document.getElementById("logAcc");
const logErrEl = document.getElementById("logErr");
const logPrecEl = document.getElementById("logPrec");
const logRecEl = document.getElementById("logRec");
const logF1El = document.getElementById("logF1");
// Tablas
const logConfTable = document.getElementById("logConf");
const tableX = document.getElementById("tableX");
const tableY = document.getElementById("tableY");
// RNA (MLP)
const mlpAccEl = document.getElementById("mlpAcc");
const mlpErrEl = document.getElementById("mlpErr");
const mlpPrecEl = document.getElementById("mlpPrec");
const mlpRecEl = document.getElementById("mlpRec");
const mlpF1El = document.getElementById("mlpF1");
const mlpConfTable = document.getElementById("mlpConf");
const mlpArchEl = document.getElementById("mlpArch");
const btnMlp = document.getElementById("btnMlp");
const btnMlpJson = document.getElementById("btnMlpJson");

function markStepDone(idx) {
  const step = document.querySelectorAll(".step")[idx];
  if (step) step.classList.add("done");
  const total = document.querySelectorAll(".step").length;
  const progress = ((idx + 1) / total) * 100;
  barEl.style.width = progress + "%";
}
function fillTable(table, rows) {
  if (!rows || rows.length === 0) return;
  const cols = Object.keys(rows[0]);
  const thead = table.querySelector("thead");
  const tbody = table.querySelector("tbody");
  thead.innerHTML = "";
  tbody.innerHTML = "";
  const trHead = document.createElement("tr");
  cols.forEach((c) => {
    const th = document.createElement("th");
    th.textContent = c;
    trHead.appendChild(th);
  });
  thead.appendChild(trHead);
  rows.forEach((r) => {
    const tr = document.createElement("tr");
    cols.forEach((c) => {
      const td = document.createElement("td");
      td.textContent = r[c];
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
}

// Pipeline completo (Logística + MLP)
async function startPipeline() {
  if (!startBtn) return;
  startBtn.disabled = true;
  errorEl.classList.add("hidden");
  resultEl.classList.add("hidden");
  stepsEl.classList.remove("hidden");
  barEl.style.width = "0%";
  document.querySelectorAll(".step").forEach((s) => s.classList.remove("done"));
  const n = Math.max(1000, parseInt(rowsInput.value || "10000", 10));
  const force = forceChk.checked ? 1 : 0;
  btnCsv.href = `/download/dataset?n=${n}&force=${force}`;
  btnJson.href = `/download/results?n=${n}&force=${force}`;
  const tick = (i, d) => new Promise((res) => setTimeout(() => { markStepDone(i); res(); }, d));
  await tick(0, 200);
  await tick(1, 200);
  try {
    await tick(2, 200);
    const resp = await fetch(`/start?n=${n}&force=${force}`);
    const data = await resp.json();
    await tick(3, 200);
    if (!data.ok) throw new Error(data.error || "Error desconocido");
    const r = data.result;

    // Resumen
    rowsEl.textContent = r.dataset_info.rows;
    colsEl.textContent = r.dataset_info.cols;
    xcolsEl.textContent = r.dataset_info.X_cols.join(", ");
    ycolsEl.textContent = r.dataset_info.Y_cols.join(", ");
    noteEl.textContent = r.dataset_info.note;

    // Métricas Logística
    const m = r.logistic.metrics;
    logAccEl.textContent = m.accuracy.toFixed(3);
    logErrEl.textContent = m.error_rate.toFixed(3);
    logPrecEl.textContent = m.precision.toFixed(3);
    logRecEl.textContent = m.recall.toFixed(3);
    logF1El.textContent = m.f1.toFixed(3);

    // Matriz de confusión (Logística)
    const cm = r.logistic.confusion_matrix; // [[tn, fp],[fn,tp]]
    {
      const tbody = logConfTable.querySelector("tbody");
      tbody.innerHTML = "";
      const row0 = document.createElement("tr");
      const th0 = document.createElement("th");
      th0.textContent = "Real: No Aprobado (0)";
      row0.appendChild(th0);
      const td00 = document.createElement("td");
      const td01 = document.createElement("td");
      td00.textContent = cm[0][0];
      td01.textContent = cm[0][1];
      row0.appendChild(td00);
      row0.appendChild(td01);

      const row1 = document.createElement("tr");
      const th1 = document.createElement("th");
      th1.textContent = "Real: Aprobado (1)";
      row1.appendChild(th1);
      const td10 = document.createElement("td");
      const td11 = document.createElement("td");
      td10.textContent = cm[1][0];
      td11.textContent = cm[1][1];
      row1.appendChild(td10);
      row1.appendChild(td11);

      tbody.appendChild(row0);
      tbody.appendChild(row1);
    }

    // Preview X/Y
    fillTable(tableX, r.preview_X);
    fillTable(tableY, r.preview_Y);

    await tick(4, 200);
    resultEl.classList.remove("hidden");
  } catch (err) {
    errorEl.textContent = "⚠️ " + err.message;
    errorEl.classList.remove("hidden");
  } finally {
    startBtn.disabled = false;
  }
}

// Solo RNA (MLP)
async function runOnlyMLP() {
  console.log("[MLP] click -> /start_mlp");
  errorEl.classList.add("hidden");
  resultEl.classList.add("hidden");
  stepsEl.classList.remove("hidden");
  barEl.style.width = "0%";
  document.querySelectorAll(".step").forEach((s) => s.classList.remove("done"));
  const n = Math.max(1000, parseInt(rowsInput.value || "10000", 10));
  const force = forceChk.checked ? 1 : 0;
  // Prepara enlace de descarga sólo MLP
  if (btnMlpJson) btnMlpJson.href = `/download/mlp_results?n=${n}&force=${force}`;
  const tick = (i, d) => new Promise((res) => setTimeout(() => { markStepDone(i); res(); }, d));
  await tick(0, 200);
  await tick(1, 200);
  try {
    await tick(2, 200);
    const resp = await fetch(`/start_mlp?n=${n}&force=${force}`);
    const data = await resp.json();
    await tick(3, 200);
    if (!data.ok) throw new Error(data.error || "Error desconocido");
    const r = data.result;

    // Resumen
    rowsEl.textContent = r.dataset_info.rows;
    colsEl.textContent = r.dataset_info.cols;
    xcolsEl.textContent = r.dataset_info.X_cols.join(", ");
    ycolsEl.textContent = r.dataset_info.Y_cols.join(", ");
    noteEl.textContent = r.dataset_info.note;

    // Preview
    fillTable(tableX, r.preview_X);
    fillTable(tableY, r.preview_Y);

    // Métricas MLP
    const mm = r.mlp.metrics;
    mlpAccEl.textContent = mm.accuracy.toFixed(3);
    mlpErrEl.textContent = mm.error_rate.toFixed(3);
    mlpPrecEl.textContent = mm.precision.toFixed(3);
    mlpRecEl.textContent = mm.recall.toFixed(3);
    mlpF1El.textContent = mm.f1.toFixed(3);

    // Matriz de confusión MLP
    const cm2 = r.mlp.confusion_matrix; // [[tn, fp],[fn,tp]]
    {
      const tbody = mlpConfTable.querySelector("tbody");
      tbody.innerHTML = "";
      const row0 = document.createElement("tr");
      const th0 = document.createElement("th");
      th0.textContent = "Real: No Aprobado (0)";
      row0.appendChild(th0);
      const td00 = document.createElement("td");
      const td01 = document.createElement("td");
      td00.textContent = cm2[0][0];
      td01.textContent = cm2[0][1];
      row0.appendChild(td00);
      row0.appendChild(td01);

      const row1 = document.createElement("tr");
      const th1 = document.createElement("th");
      th1.textContent = "Real: Aprobado (1)";
      row1.appendChild(th1);
      const td10 = document.createElement("td");
      const td11 = document.createElement("td");
      td10.textContent = cm2[1][0];
      td11.textContent = cm2[1][1];
      row1.appendChild(td10);
      row1.appendChild(td11);

      tbody.appendChild(row0);
      tbody.appendChild(row1);
    }

    // Arquitectura MLP
    const arch = r.mlp.architecture;
    mlpArchEl.textContent = JSON.stringify(arch, null, 2);

    await tick(4, 200);
    resultEl.classList.remove("hidden");
    document.getElementById("mlpArch").scrollIntoView({ behavior: "smooth" });
  } catch (err) {
    errorEl.textContent = "⚠️ " + err.message;
    errorEl.classList.remove("hidden");
  }
}

// Conexión de listeners cuando el DOM esté listo
document.addEventListener("DOMContentLoaded", () => {
  const btnMlp = document.getElementById("btnMlp");
  const startBtn = document.getElementById("startBtn");
  if (startBtn) {
    startBtn.addEventListener("click", (e) => { e.preventDefault(); startPipeline(); });
  }
  if (btnMlp) {
    btnMlp.addEventListener("click", (e) => { e.preventDefault(); runOnlyMLP(); });
  }
});
