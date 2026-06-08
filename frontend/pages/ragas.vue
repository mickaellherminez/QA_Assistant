<script setup lang="ts">
import type { RagasResponse, RagasRunResponse } from '~/types/api'

type ReportKey = 'reranker' | 'baseline' | 'v2'

const selectedReport = ref<ReportKey>('reranker')
const selectedModel = ref('gpt-4o-mini')
const selectedEmbeddingModel = ref('text-embedding-3-small')
const maxSamples = ref(5)

const isRunning = ref(false)
const runError = ref('')
const liveRun = ref<RagasRunResponse | null>(null)

const {
  data,
  pending,
  error,
  refresh
} = await useFetch<RagasResponse>('/api/ragas', {
  query: computed(() => ({
    report: selectedReport.value
  }))
})

const reportOptions = computed(() => {
  const ordered: ReportKey[] = ['reranker', 'baseline', 'v2']
  const dynamic = data.value?.available_reports ?? []
  const merged = [...new Set([...ordered, ...dynamic])]
  return merged
})

const effectiveData = computed<RagasResponse | null>(() => {
  if (liveRun.value) {
    return {
      selected_report: liveRun.value.report,
      report_file: liveRun.value.report_file || 'live-run',
      available_reports: liveRun.value.available_reports,
      scores: liveRun.value.scores,
      global_score: liveRun.value.global_score,
      n_samples: liveRun.value.n_samples,
      model: liveRun.value.model
    }
  }
  return data.value ?? null
})

const globalScore = computed(() => {
  const score = effectiveData.value?.global_score
  return typeof score === 'number' ? score : 0
})

const gaugeStyle = computed(() => {
  const clamped = Math.max(0, Math.min(1, globalScore.value))
  const percent = clamped * 100
  return {
    background: `conic-gradient(var(--accent) ${percent}%, rgba(217, 204, 180, 0.38) ${percent}% 100%)`
  }
})

function metricLabel(metricName: string): string {
  const labels: Record<string, string> = {
    answer_relevancy: 'Pertinence réponse',
    context_precision: 'Précision contexte',
    context_recall: 'Rappel contexte',
    faithfulness: 'Fidélité sources',
    response_relevancy: 'Pertinence réponse'
  }
  return labels[metricName] || metricName.replaceAll('_', ' ')
}

function metricTone(value: number): 'good' | 'warn' | 'bad' {
  if (value >= 0.75) return 'good'
  if (value >= 0.55) return 'warn'
  return 'bad'
}

const scoreRows = computed(() => {
  if (!effectiveData.value?.scores) return []

  return Object.entries(effectiveData.value.scores)
    .filter(([, value]): value is number => typeof value === 'number')
    .map(([name, value]) => ({
      name,
      label: metricLabel(name),
      value,
      percent: Math.max(0, Math.min(1, value)) * 100,
      tone: metricTone(value)
    }))
    .sort((a, b) => b.value - a.value)
})

const runSummary = computed(() => {
  if (!liveRun.value) return ''
  return `Run terminé en ${liveRun.value.duration_ms} ms.`
})

watch(selectedReport, () => {
  liveRun.value = null
})

async function runRagasLive() {
  isRunning.value = true
  runError.value = ''

  try {
    const payload = await $fetch<RagasRunResponse>('/api/ragas/run', {
      method: 'POST',
      body: {
        report: selectedReport.value,
        persist_report: true,
        model: selectedModel.value.trim() || undefined,
        embedding_model: selectedEmbeddingModel.value.trim() || undefined,
        max_samples: maxSamples.value
      }
    })

    liveRun.value = payload
    if (payload.persisted) {
      await refresh()
    }
  } catch (err) {
    runError.value = err instanceof Error ? err.message : 'Erreur pendant le run RAGAS.'
  } finally {
    isRunning.value = false
  }
}

</script>

<template>
  <section class="page-wrap">
    <div class="page-head">
      <h1>RAGAS Dashboard</h1>
      <p>Choix du modèle, benchmark en direct, et visualisation graphique des scores qualité RAG.</p>
    </div>

    <div class="toolbar card ragas-toolbar ragas-panel">
      <div class="toolbar-group">
        <label for="report">Rapport cible</label>
        <select id="report" v-model="selectedReport">
          <option v-for="reportName in reportOptions" :key="reportName" :value="reportName">
            {{ reportName }}
          </option>
        </select>
      </div>

      <div class="toolbar-group">
        <label for="model">Modèle juge</label>
        <input id="model" v-model="selectedModel" type="text" placeholder="gpt-4o-mini" />
      </div>

      <div class="toolbar-group">
        <label for="embedding">Embeddings</label>
        <input id="embedding" v-model="selectedEmbeddingModel" type="text" placeholder="text-embedding-3-small" />
      </div>

      <div class="toolbar-group">
        <label for="samples">Max samples</label>
        <input id="samples" v-model.number="maxSamples" type="number" min="1" max="25" />
      </div>

      <div class="toolbar-actions">
        <button class="btn" :disabled="isRunning" @click="runRagasLive()">
          <span v-if="isRunning">Run en cours...</span>
          <span v-else>Lancer benchmark</span>
        </button>
      </div>
    </div>

    <p v-if="pending" class="state-line">Chargement du rapport RAGAS...</p>
    <p v-else-if="error" class="state-line state-error">Erreur RAGAS: {{ error.message }}</p>
    <p v-if="runError" class="state-line state-error">{{ runError }}</p>
    <p v-if="runSummary" class="state-line">{{ runSummary }}</p>

    <div v-if="effectiveData" class="ragas-grid">
      <div class="ragas-kpi-row">
        <article class="card kpi-card ragas-kpi-card score-gauge-card">
          <p class="kpi-label">Global Score</p>
          <div class="score-gauge">
            <div class="score-gauge-ring" :style="gaugeStyle">
              <div class="score-gauge-inner">
                <strong>{{ globalScore.toFixed(3) }}</strong>
              </div>
            </div>
          </div>
        </article>

        <article class="card kpi-card ragas-kpi-card">
          <p class="kpi-label">Rapport</p>
          <p class="kpi-value mono">{{ effectiveData.selected_report }}</p>
          <p class="kpi-meta">Fichier: <code>{{ effectiveData.report_file }}</code></p>
        </article>

        <article class="card kpi-card ragas-kpi-card">
          <p class="kpi-label">Model</p>
          <p class="kpi-value mono">{{ effectiveData.model ?? 'N/A' }}</p>
          <p class="kpi-meta">Embeddings: <code>{{ selectedEmbeddingModel }}</code></p>
        </article>

        <article class="card kpi-card ragas-kpi-card">
          <p class="kpi-label">Samples</p>
          <p class="kpi-value">{{ effectiveData.n_samples ?? 'N/A' }}</p>
        </article>
      </div>

      <article class="card kpi-card ragas-score-card ragas-panel">
        <p class="kpi-label">Scores détaillés</p>
        <ul class="metric-bars">
          <li v-for="metric in scoreRows" :key="metric.name">
            <div class="metric-head">
              <span class="mono">{{ metric.label }}</span>
              <strong>{{ metric.value.toFixed(3) }}</strong>
            </div>
            <div class="metric-track">
              <div class="metric-fill" :class="`tone-${metric.tone}`" :style="{ width: `${metric.percent}%` }" />
            </div>
          </li>
        </ul>
      </article>
    </div>
  </section>
</template>
