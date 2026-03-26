//
//  SetupView.swift
//  CoreMetric
//
//  Step-by-step setup wizard. Shows live terminal output from the Python
//  pipeline and guides the user from zero to a trained model.
//

import SwiftUI

struct SetupView: View {
    @ObservedObject var orchestrator: SetupOrchestrator
    let onComplete: () -> Void

    @Environment(\.dismiss) private var dismiss

    // Tracks the last non-failed phase so step indicators stay accurate on error
    @State private var displayPhase: SetupOrchestrator.Phase = .idle

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            HStack(alignment: .top, spacing: 0) {
                stepsPanel
                    .frame(width: 192)
                Divider()
                outputPanel
            }
            Divider()
            actionBar
        }
        .frame(width: 580, height: 500)
        .onChange(of: orchestrator.phase) { phase in
            if case .failed = phase {} else { displayPhase = phase }
            if phase == .complete { onComplete() }
        }
    }

    // MARK: - Header

    private var header: some View {
        HStack(alignment: .center) {
            Image(systemName: "brain")
                .font(.title2)
                .foregroundColor(.accentColor)
            VStack(alignment: .leading, spacing: 1) {
                Text("CoreMetric Setup")
                    .font(.headline)
                Text("Train a model of your Mac's normal behavior")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            Spacer()
        }
        .padding(.horizontal)
        .padding(.vertical, 12)
    }

    // MARK: - Steps Panel

    private var stepsPanel: some View {
        VStack(alignment: .leading, spacing: 0) {
            ForEach(Array(steps.enumerated()), id: \.offset) { i, step in
                stepRow(index: i, title: step.title, activePhase: step.activePhase)
                if i < steps.count - 1 {
                    Rectangle()
                        .fill(Color.gray.opacity(0.15))
                        .frame(width: 1, height: 12)
                        .padding(.leading, 27)
                }
            }

            Spacer()

            // Collect live stats block
            if orchestrator.phase == .collecting {
                collectStatsBlock
                    .padding(.horizontal, 12)
                    .padding(.bottom, 12)
            }

            // Training progress block
            if orchestrator.phase == .training {
                trainingProgressBlock
                    .padding(.horizontal, 12)
                    .padding(.bottom, 12)
            }
        }
        .padding(.vertical, 12)
    }

    private struct StepSpec {
        let title: String
        let activePhase: SetupOrchestrator.Phase
    }

    private let steps: [StepSpec] = [
        StepSpec(title: "Find Python",        activePhase: .findingPython),
        StepSpec(title: "Set Up Environment", activePhase: .creatingEnv),
        StepSpec(title: "Install Libraries",  activePhase: .installingDeps),
        StepSpec(title: "Collect Baseline",   activePhase: .readyToCollect),
        StepSpec(title: "Train Model",        activePhase: .training),
        StepSpec(title: "Activate",           activePhase: .complete),
    ]

    private func stepRow(index: Int, title: String, activePhase: SetupOrchestrator.Phase) -> some View {
        let (isDone, isActive) = stepState(activePhase: activePhase, index: index)
        return HStack(spacing: 10) {
            stepBadge(number: index + 1, isDone: isDone, isActive: isActive)
            Text(title)
                .font(.caption)
                .fontWeight(isActive ? .semibold : .regular)
                .foregroundColor(isDone || isActive ? .primary : .secondary)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
    }

    @ViewBuilder
    private func stepBadge(number: Int, isDone: Bool, isActive: Bool) -> some View {
        ZStack {
            Circle()
                .fill(isDone ? Color.green : (isActive ? Color.accentColor : Color.gray.opacity(0.18)))
                .frame(width: 22, height: 22)
            if isDone {
                Image(systemName: "checkmark")
                    .font(.system(size: 9, weight: .bold))
                    .foregroundColor(.white)
            } else if isActive {
                ProgressView()
                    .scaleEffect(0.45)
                    .colorScheme(.dark)
            } else {
                Text("\(number)")
                    .font(.system(size: 9, weight: .semibold))
                    .foregroundColor(.gray)
            }
        }
    }

    /// Returns (isDone, isActive) for a step given its representative phase.
    private func stepState(activePhase: SetupOrchestrator.Phase, index: Int) -> (Bool, Bool) {
        let order: [SetupOrchestrator.Phase] = [
            .idle, .findingPython, .checkingPackages, .creatingEnv, .installingDeps,
            .readyToCollect, .collecting, .training, .complete
        ]
        let currentIdx = order.firstIndex(of: displayPhase) ?? 0
        let activeIdx  = order.firstIndex(of: activePhase) ?? 0

        // "Activate" step: done only when phase is .complete (no "next" phase exists)
        if activePhase == .complete {
            return (displayPhase == .complete, false)
        }

        // "Find Python" step: active during both .findingPython and .checkingPackages
        let isFindStep    = (activePhase == .findingPython)
        // "Collect Baseline" step: active during both .readyToCollect and .collecting
        let isCollectStep = (activePhase == .readyToCollect)

        let doneThreshold: Int
        if isFindStep {
            doneThreshold = order.firstIndex(of: .creatingEnv) ?? (activeIdx + 1)
        } else if isCollectStep {
            doneThreshold = order.firstIndex(of: .training) ?? (activeIdx + 1)
        } else {
            doneThreshold = activeIdx + 1
        }

        let isDone = currentIdx >= doneThreshold
        let checkingIdx  = order.firstIndex(of: .checkingPackages) ?? -1
        let collectingIdx = order.firstIndex(of: .collecting) ?? -1
        let isActive = !isDone && (
            currentIdx == activeIdx
            || (isFindStep    && currentIdx == checkingIdx)
            || (isCollectStep && currentIdx == collectingIdx)
        )
        return (isDone, isActive)
    }

    private var collectStatsBlock: some View {
        VStack(alignment: .leading, spacing: 6) {
            Label(formatElapsed(orchestrator.collectElapsed), systemImage: "record.circle.fill")
                .font(.caption)
                .foregroundColor(.red)

            if !orchestrator.liveStats.isEmpty {
                Text(orchestrator.liveStats
                        .replacingOccurrences(of: "[ REC ] ", with: "")
                        .trimmingCharacters(in: .whitespaces))
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundColor(.secondary)
                    .lineLimit(1)
            }

            if orchestrator.collectElapsed < 300 {
                Text("Collect at least 30 min for best accuracy")
                    .font(.caption2)
                    .foregroundColor(.orange)
            } else {
                Text("Great — ready to train when you are")
                    .font(.caption2)
                    .foregroundColor(.green)
            }
        }
        .padding(8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 6).fill(Color.red.opacity(0.07)))
    }

    private var trainingProgressBlock: some View {
        VStack(alignment: .leading, spacing: 5) {
            ProgressView(value: orchestrator.trainProgress)
                .progressViewStyle(.linear)
                .tint(.accentColor)
            Text("Epoch \(Int(orchestrator.trainProgress * 300))/300 — \(Int(orchestrator.trainProgress * 100))%")
                .font(.caption2)
                .foregroundColor(.secondary)
        }
    }

    // MARK: - Output Panel

    private var outputPanel: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 1) {
                    ForEach(Array(orchestrator.outputLines.enumerated()), id: \.offset) { _, line in
                        Text(line)
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundColor(lineColor(for: line))
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(.horizontal, 10)
                    }
                    Color.clear.frame(height: 1).id("log-bottom")
                }
                .padding(.vertical, 8)
            }
            .background(Color(nsColor: .textBackgroundColor))
            .onChange(of: orchestrator.outputLines.count) { _ in
                withAnimation(.linear(duration: 0.1)) {
                    proxy.scrollTo("log-bottom")
                }
            }
        }
    }

    // MARK: - Action Bar

    private var actionBar: some View {
        HStack(spacing: 12) {
            actionContent
        }
        .padding(.horizontal)
        .frame(height: 56)
    }

    @ViewBuilder
    private var actionContent: some View {
        switch orchestrator.phase {
        case .idle:
            Spacer()
            Button("Begin Setup") { orchestrator.start() }
                .buttonStyle(.borderedProminent)

        case .findingPython, .checkingPackages, .creatingEnv, .installingDeps:
            ProgressView().scaleEffect(0.75)
            Text(phaseLabel)
                .font(.caption)
                .foregroundColor(.secondary)
            Spacer()

        case .readyToCollect:
            VStack(alignment: .leading, spacing: 1) {
                Text("Ready to record")
                    .font(.caption).fontWeight(.medium)
                Text("Click Start and use your Mac normally")
                    .font(.caption2).foregroundColor(.secondary)
            }
            Spacer()
            Button("Start Recording") { orchestrator.startCollecting() }
                .buttonStyle(.borderedProminent)

        case .collecting:
            VStack(alignment: .leading, spacing: 1) {
                Text("Recording…")
                    .font(.caption).fontWeight(.medium)
                Text("Stop whenever you have enough data (30+ min is ideal)")
                    .font(.caption2).foregroundColor(.secondary)
            }
            Spacer()
            Button("Stop & Train") { orchestrator.stopCollectingAndTrain() }
                .buttonStyle(.borderedProminent)
                .disabled(orchestrator.collectElapsed < 60) // require at least 1 min

        case .training:
            ProgressView().scaleEffect(0.75)
            Text("Training model…")
                .font(.caption).foregroundColor(.secondary)
            Spacer()

        case .complete:
            Label("Model trained and active", systemImage: "checkmark.circle.fill")
                .font(.caption).foregroundColor(.green)
            Spacer()
            Button("Start Monitoring") { dismiss() }
                .buttonStyle(.borderedProminent)

        case .failed(let msg):
            VStack(alignment: .leading, spacing: 2) {
                Text("Setup failed")
                    .font(.caption).fontWeight(.semibold).foregroundColor(.red)
                Text(msg)
                    .font(.caption2).foregroundColor(.secondary)
                    .lineLimit(2)
            }
            Spacer()
            Button("Try Again") { orchestrator.start() }
                .buttonStyle(.borderedProminent)
        }
    }

    // MARK: - Helpers

    private var phaseLabel: String {
        switch orchestrator.phase {
        case .findingPython:     return "Finding Python…"
        case .checkingPackages:  return "Checking installed packages…"
        case .creatingEnv:       return "Creating virtual environment…"
        case .installingDeps:    return "Installing ML libraries…"
        default:                 return "Working…"
        }
    }

    private func lineColor(for line: String) -> Color {
        let l = line.lowercased()
        if l.contains("error") || l.contains("failed") || l.contains("critical") { return .red }
        if l.contains("warning") || l.contains("warn")                            { return .orange }
        if l.contains("done") || l.contains("success") || l.hasPrefix("✓")        { return .green }
        return Color(nsColor: .labelColor)
    }

    private func formatElapsed(_ t: TimeInterval) -> String {
        let m = Int(t) / 60
        let s = Int(t) % 60
        return String(format: "%02d:%02d", m, s)
    }
}

#Preview {
    let o = SetupOrchestrator()
    return SetupView(orchestrator: o, onComplete: {})
}
