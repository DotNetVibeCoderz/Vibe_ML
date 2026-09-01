{{/*
Naming helpers, following the conventions `helm create` establishes so that this chart behaves
the way anyone reading it expects.
*/}}

{{- define "localgen.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "localgen.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "localgen.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "localgen.labels" -}}
helm.sh/chart: {{ include "localgen.chart" . }}
{{ include "localgen.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: localgen
{{- end -}}

{{- define "localgen.selectorLabels" -}}
app.kubernetes.io/name: {{ include "localgen.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/component: inference
{{- end -}}

{{- define "localgen.web.selectorLabels" -}}
app.kubernetes.io/name: {{ include "localgen.name" . }}-web
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/component: web
{{- end -}}

{{- define "localgen.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "localgen.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{/*
The image the inference pod runs. A GPU deployment needs the CUDA build: the CPU image cannot use
an accelerator even when the device plugin has attached one, and the failure is silent — the pod
runs, just slowly.
*/}}
{{- define "localgen.image" -}}
{{- if .Values.gpu.enabled -}}
{{- printf "%s:%s" .Values.gpu.image.repository (default .Chart.AppVersion .Values.gpu.image.tag) -}}
{{- else -}}
{{- printf "%s:%s" .Values.image.repository (default .Chart.AppVersion .Values.image.tag) -}}
{{- end -}}
{{- end -}}

{{- define "localgen.secretName" -}}
{{- if .Values.auth.existingSecret -}}
{{- .Values.auth.existingSecret -}}
{{- else -}}
{{- printf "%s-auth" (include "localgen.fullname" .) -}}
{{- end -}}
{{- end -}}

{{/*
Whether a Secret has to be created for this release. An existing secret is used as-is; otherwise
one is needed as soon as any key is in play.
*/}}
{{- define "localgen.createSecret" -}}
{{- if .Values.auth.existingSecret -}}
false
{{- else if or .Values.auth.enabled .Values.auth.keys -}}
true
{{- else -}}
false
{{- end -}}
{{- end -}}
