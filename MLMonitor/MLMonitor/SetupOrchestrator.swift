//
//  SetupOrchestrator.swift
//  CoreMetric
//
//  Manages the full model training pipeline — find Python, check/install deps,
//  collect data, train — all in-process with live output streamed to the UI.
//

import Foundation
import Combine

class SetupOrchestrator: ObservableObject {

    enum Phase: Equatable {
        case idle
        case findingPython
        case checkingPackages
        case creatingEnv
        case installingDeps
        case readyToCollect
        case collecting
        case training
        case complete
        case failed(String)

        static func == (lhs: Phase, rhs: Phase) -> Bool {
            switch (lhs, rhs) {
            case (.idle, .idle), (.findingPython, .findingPython),
                 (.checkingPackages, .checkingPackages), (.creatingEnv, .creatingEnv),
                 (.installingDeps, .installingDeps), (.readyToCollect, .readyToCollect),
                 (.collecting, .collecting), (.training, .training), (.complete, .complete):
                return true
            case let (.failed(a), .failed(b)): return a == b
            default: return false
            }
        }
    }

    @Published var phase: Phase = .idle
    @Published var outputLines: [String] = []
    @Published var liveStats: String = ""
    @Published var collectElapsed: TimeInterval = 0
    @Published var trainProgress: Double = 0

    let supportDir: URL

    // The Python binary we'll use to run the scripts.
    // Set to a system Python if packages are already there, otherwise venv Python.
    private var scriptPython: String?

    private var collectProcess: Process?
    private var collectTimer: Timer?
    private var collectStart: Date?
    // Retained references so ARC doesn't release processes mid-run
    private var activeProcess: Process?

    // MARK: - Init

    init() {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        supportDir = base.appendingPathComponent("CoreMetric")
        try? FileManager.default.createDirectory(
            at: supportDir.appendingPathComponent("scripts"), withIntermediateDirectories: true)
        try? FileManager.default.createDirectory(
            at: supportDir.appendingPathComponent("data/raw"), withIntermediateDirectories: true)
    }

    // MARK: - Public API

    func start() {
        outputLines = []
        trainProgress = 0
        liveStats = ""
        scriptPython = nil
        writeScripts()
        findPython()
    }

    func startCollecting() {
        guard let python = scriptPython else {
            phase = .failed("No Python runtime found. Run setup again.")
            return
        }
        phase = .collecting
        collectStart = Date()
        collectElapsed = 0
        liveStats = ""

        collectTimer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in
            guard let self, let start = self.collectStart else { return }
            self.collectElapsed = Date().timeIntervalSince(start)
        }

        log("Starting data collection…")
        log("Use your Mac normally — browse, code, watch video.")
        log("The longer you record, the more accurate the model.")

        let script = supportDir.appendingPathComponent("scripts/collect.py").path
        let proc = makeProcess(exec: python, args: [script])
        let reader = OutputReader { [weak self] line in
            guard let self else { return }
            if line.contains("[ REC ]") {
                self.liveStats = line
                if self.outputLines.last?.contains("[ REC ]") == true {
                    self.outputLines[self.outputLines.count - 1] = line
                } else {
                    self.appendLog(line)
                }
            } else if !line.isEmpty {
                self.appendLog(line)
            }
        }
        attachOutput(reader, to: proc)
        launch(proc) { _ in /* collect runs until user stops */ }
        collectProcess = proc
    }

    func stopCollectingAndTrain() {
        collectTimer?.invalidate()
        collectTimer = nil
        collectProcess?.interrupt()   // SIGINT → graceful Python save
        collectProcess = nil
        trainModel()
    }

    func cancel() {
        collectTimer?.invalidate()
        collectProcess?.terminate()
        activeProcess?.terminate()
        phase = .idle
    }

    // MARK: - Pipeline

    private func findPython() {
        phase = .findingPython
        log("Looking for Python 3…")

        let candidates = [
            "/opt/homebrew/bin/python3",
            "/usr/local/bin/python3",
            "/usr/bin/python3",
        ]

        tryPythonCandidates(candidates, index: 0)
    }

    /// Try each candidate in order, verify it actually executes (not just a file-exists check,
    /// which passes for the macOS stub that blocks waiting for Xcode CLT install).
    private func tryPythonCandidates(_ candidates: [String], index: Int) {
        guard index < candidates.count else {
            phase = .failed("Python 3 not found.\nInstall it from python.org or via Homebrew.")
            return
        }

        let path = candidates[index]
        guard FileManager.default.isExecutableFile(atPath: path) else {
            tryPythonCandidates(candidates, index: index + 1)
            return
        }

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: path)
        proc.arguments = ["--version"]
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError  = pipe

        proc.terminationHandler = { [weak self] p in
            let out = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            DispatchQueue.main.async {
                if p.terminationStatus == 0, out.lowercased().hasPrefix("python") {
                    self?.log("Found \(out) at \(path)")
                    self?.checkPackages(python: path)
                } else {
                    // This binary didn't respond as expected — try next
                    self?.tryPythonCandidates(candidates, index: index + 1)
                }
            }
        }

        do {
            try proc.run()
        } catch {
            tryPythonCandidates(candidates, index: index + 1)
        }
    }

    /// Ask Python itself whether all required packages are importable.
    /// This is the only reliable check — filesystem heuristics miss partial installs.
    private func checkPackages(python: String) {
        phase = .checkingPackages
        log("Checking for required packages in \(python)…")

        let script = """
import importlib.util, sys
pkgs = {"psutil":"psutil","pandas":"pandas","numpy":"numpy",\
"scikit-learn":"sklearn","torch":"torch","coremltools":"coremltools"}
missing = [k for k,v in pkgs.items() if importlib.util.find_spec(v) is None]
if missing:
    print("MISSING:" + ",".join(missing), flush=True)
    sys.exit(1)
print("ALL_OK", flush=True)
"""

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: python)
        proc.arguments = ["-c", script]
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError  = Pipe()   // suppress ImportWarnings etc.

        proc.terminationHandler = { [weak self] p in
            let out = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            DispatchQueue.main.async {
                guard let self else { return }
                if out == "ALL_OK" {
                    self.log("All required packages found — skipping installation")
                    self.scriptPython = python
                    self.phase = .readyToCollect
                } else {
                    // Report what's missing, then decide where to install
                    if out.hasPrefix("MISSING:") {
                        self.log("Missing: \(out.dropFirst("MISSING:".count))")
                    }
                    // Before creating a new venv, check if our existing venv is already good
                    self.resolveEnvironment(systemPython: python)
                }
            }
        }

        do {
            try proc.run()
        } catch {
            // Can't run the check — treat as missing and proceed to env resolution
            DispatchQueue.main.async { self.resolveEnvironment(systemPython: python) }
        }
    }

    /// Decide whether to use, fix, or create a venv.
    private func resolveEnvironment(systemPython: String) {
        if let vpy = venvPython {
            // A venv already exists — check if it has what we need
            log("Checking existing virtual environment…")
            checkPackagesInVenv(venvPython: vpy, systemPython: systemPython)
        } else {
            createEnv(systemPython: systemPython)
        }
    }

    private func checkPackagesInVenv(venvPython: String, systemPython: String) {
        let script = """
import importlib.util, sys
pkgs = {"psutil":"psutil","pandas":"pandas","numpy":"numpy",\
"scikit-learn":"sklearn","torch":"torch","coremltools":"coremltools"}
missing = [k for k,v in pkgs.items() if importlib.util.find_spec(v) is None]
if missing:
    print("MISSING:" + ",".join(missing), flush=True)
    sys.exit(1)
print("ALL_OK", flush=True)
"""
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: venvPython)
        proc.arguments = ["-c", script]
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError  = Pipe()

        proc.terminationHandler = { [weak self] p in
            let out = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            DispatchQueue.main.async {
                guard let self else { return }
                if out == "ALL_OK" {
                    self.log("Existing virtual environment has all packages")
                    self.scriptPython = venvPython
                    self.phase = .readyToCollect
                } else {
                    // Venv exists but is incomplete — install into it
                    if out.hasPrefix("MISSING:") {
                        self.log("Venv missing: \(out.dropFirst("MISSING:".count)) — installing…")
                    }
                    self.installDeps()
                }
            }
        }

        do {
            try proc.run()
        } catch {
            DispatchQueue.main.async { self.createEnv(systemPython: systemPython) }
        }
    }

    private func createEnv(systemPython: String) {
        phase = .creatingEnv
        log("Creating isolated Python environment…")

        let venvPath = supportDir.appendingPathComponent("venv").path
        runBlocking(exec: systemPython, args: ["-m", "venv", venvPath]) { [weak self] code in
            if code == 0 {
                self?.log("Environment created")
                self?.installDeps()
            } else {
                self?.phase = .failed("Could not create Python environment (exit \(code)).\nMake sure Python 3 is fully installed.")
            }
        }
    }

    private func installDeps() {
        guard let pip = resolvedPip else {
            phase = .failed("pip not found in virtual environment.")
            return
        }
        phase = .installingDeps
        log("Installing ML libraries into virtual environment…")
        log("First run: ~1–2 GB download, may take several minutes.")

        let req = supportDir.appendingPathComponent("requirements.txt").path

        // --no-progress-bar: disables \r-based progress so output streams cleanly
        runStreaming(exec: pip, args: ["install", "-r", req, "--no-progress-bar"]) { [weak self] line in
            let t = line.trimmingCharacters(in: .whitespaces)
            guard !t.isEmpty else { return }
            // Show substantive pip output, skip internal pip noise
            let skip = t.hasPrefix("WARNING: pip") || t.hasPrefix("Notice:")
            if !skip { self?.appendLog(t) }
        } onComplete: { [weak self] code in
            guard let self else { return }
            if code == 0 {
                log("Libraries installed")
                if let vpy = venvPython {
                    scriptPython = vpy
                    phase = .readyToCollect
                } else {
                    phase = .failed("Virtual environment Python not found after installation.")
                }
            } else {
                phase = .failed("Installation failed (exit \(code)).\nCheck internet connection and try again.")
            }
        }
    }

    private func trainModel() {
        guard let python = scriptPython else {
            phase = .failed("No Python runtime available. Run setup again.")
            return
        }
        phase = .training
        trainProgress = 0
        log("\nTraining model on collected data…")

        let script = supportDir.appendingPathComponent("scripts/train_coreml.py").path

        runStreaming(exec: python, args: [script]) { [weak self] line in
            let t = line.trimmingCharacters(in: .whitespaces)
            guard !t.isEmpty else { return }
            self?.appendLog(t)
            if t.contains("Epoch "), let n = Self.parseEpoch(t) {
                self?.trainProgress = min(1.0, Double(n) / 300.0)
            }
        } onComplete: { [weak self] code in
            guard let self else { return }
            if code == 0, FileManager.default.fileExists(atPath: InferenceEngine.modelURL.path) {
                log("Model saved and ready")
                phase = .complete
            } else if code == 0 {
                phase = .failed("Training finished but model file was not found at expected path.")
            } else {
                phase = .failed("Training failed (exit \(code)).\nMake sure you collected enough data.")
            }
        }
    }

    // MARK: - Helpers

    private var venvPython: String? {
        let p = supportDir.appendingPathComponent("venv/bin/python3").path
        return FileManager.default.fileExists(atPath: p) ? p : nil
    }

    private var resolvedPip: String? {
        let pip3 = supportDir.appendingPathComponent("venv/bin/pip3").path
        if FileManager.default.fileExists(atPath: pip3) { return pip3 }
        let pip  = supportDir.appendingPathComponent("venv/bin/pip").path
        if FileManager.default.fileExists(atPath: pip)  { return pip }
        return nil
    }

    private func writeScripts() {
        let dir = supportDir.appendingPathComponent("scripts")
        try? PythonScripts.collect.write(
            to: dir.appendingPathComponent("collect.py"), atomically: true, encoding: .utf8)
        try? PythonScripts.train.write(
            to: dir.appendingPathComponent("train_coreml.py"), atomically: true, encoding: .utf8)
        try? PythonScripts.requirements.write(
            to: supportDir.appendingPathComponent("requirements.txt"), atomically: true, encoding: .utf8)
    }

    private func log(_ text: String) { appendLog(text) }

    private func appendLog(_ line: String) {
        outputLines.append(line)
        if outputLines.count > 500 { outputLines.removeFirst(100) }
    }

    private func makeProcess(exec: String, args: [String]) -> Process {
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: exec)
        proc.arguments = args
        proc.currentDirectoryURL = supportDir
        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"
        proc.environment = env
        return proc
    }

    private func attachOutput(_ reader: OutputReader, to proc: Process) {
        let out = Pipe()
        let err = Pipe()
        out.fileHandleForReading.readabilityHandler = { reader.receive($0.availableData) }
        err.fileHandleForReading.readabilityHandler = { reader.receive($0.availableData) }
        proc.standardOutput = out
        proc.standardError  = err
    }

    /// Launch a process, retaining it until it terminates. Calls completion on main thread.
    private func launch(_ proc: Process, completion: @escaping (Int32) -> Void) {
        activeProcess = proc
        proc.terminationHandler = { [weak self] p in
            DispatchQueue.main.async {
                self?.activeProcess = nil
                completion(p.terminationStatus)
            }
        }
        do {
            try proc.run()
        } catch {
            DispatchQueue.main.async {
                self.activeProcess = nil
                completion(-1)
            }
        }
    }

    /// Blocking process with full output logged.
    private func runBlocking(exec: String, args: [String], onComplete: @escaping (Int32) -> Void) {
        let proc = makeProcess(exec: exec, args: args)
        let reader = OutputReader { [weak self] line in
            guard !line.isEmpty else { return }
            self?.appendLog(line)
        }
        attachOutput(reader, to: proc)
        launch(proc, completion: onComplete)
    }

    /// Streaming process — calls onLine per line, then onComplete.
    private func runStreaming(exec: String, args: [String],
                              onLine: @escaping (String) -> Void,
                              onComplete: @escaping (Int32) -> Void) {
        let proc = makeProcess(exec: exec, args: args)
        let reader = OutputReader(onLine: onLine)
        attachOutput(reader, to: proc)
        launch(proc, completion: onComplete)
    }

    private static func parseEpoch(_ line: String) -> Int? {
        let tokens = line.split(separator: " ").map(String.init)
        guard let idx = tokens.firstIndex(of: "Epoch"), idx + 1 < tokens.count else { return nil }
        return Int(tokens[idx + 1].trimmingCharacters(in: CharacterSet(charactersIn: ":")))
    }
}

// MARK: - OutputReader

/// Thread-safe line splitter. Buffers incoming bytes on a serial queue,
/// splits on \n and \r, dispatches each complete line to the main thread.
private class OutputReader {
    private let queue = DispatchQueue(label: "CoreMetric.OutputReader.\(UUID())")
    private var buffer = ""
    private let onLine: (String) -> Void

    init(onLine: @escaping (String) -> Void) {
        self.onLine = onLine
    }

    func receive(_ data: Data) {
        guard !data.isEmpty, let str = String(data: data, encoding: .utf8) else { return }
        queue.async { [weak self] in
            guard let self else { return }
            self.buffer += str
            var parts = self.buffer.components(separatedBy: CharacterSet(charactersIn: "\n\r"))
            self.buffer = parts.removeLast()
            let lines = parts.map { $0.trimmingCharacters(in: .whitespaces) }
            DispatchQueue.main.async { lines.forEach { self.onLine($0) } }
        }
    }
}
