{{- define "its-hub.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "its-hub.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name (include "its-hub.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{- /*
Per-resource names. The Deployment keeps the bare fullname as the primary object;
other resources carry a short kind suffix. Each is truncated to 63 characters.
*/ -}}
{{- define "its-hub.service.name" -}}
{{- printf "%s-svc" (include "its-hub.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- define "its-hub.configmap.name" -}}
{{- printf "%s-config" (include "its-hub.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- define "its-hub.route.name" -}}
{{- printf "%s-route" (include "its-hub.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- define "its-hub.ingress.name" -}}
{{- printf "%s-ingress" (include "its-hub.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- define "its-hub.hpa.name" -}}
{{- printf "%s-hpa" (include "its-hub.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- define "its-hub.pdb.name" -}}
{{- printf "%s-pdb" (include "its-hub.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- define "its-hub.networkpolicy.name" -}}
{{- printf "%s-netpol" (include "its-hub.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- define "its-hub.selectorLabels" -}}
app.kubernetes.io/name: {{ include "its-hub.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "its-hub.labels" -}}
{{ include "its-hub.selectorLabels" . }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "its-hub.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (printf "%s-sa" (include "its-hub.fullname" .) | trunc 63 | trimSuffix "-") .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{- define "its-hub.image" -}}
{{- if .Values.image.digest -}}
{{- printf "%s@%s" .Values.image.repository .Values.image.digest -}}
{{- else -}}
{{- printf "%s:%s" .Values.image.repository (required "Set image.tag or image.digest to a published image containing startup configuration and /ready support" .Values.image.tag) -}}
{{- end -}}
{{- end -}}

{{- /*
Resolve the external exposure kind. Empty when disabled. With kind=auto (the
default) it renders a Route on clusters exposing route.openshift.io/v1 and an
Ingress everywhere else; kind=route|ingress forces one explicitly.
*/ -}}
{{- define "its-hub.exposeKind" -}}
{{- if .Values.expose.enabled -}}
{{- $kind := .Values.expose.kind | default "auto" -}}
{{- if eq $kind "auto" -}}
{{- if .Capabilities.APIVersions.Has "route.openshift.io/v1" -}}route{{- else -}}ingress{{- end -}}
{{- else -}}
{{- $kind -}}
{{- end -}}
{{- end -}}
{{- end -}}
