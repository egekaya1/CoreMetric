//
//  ProcessInspector.swift
//  CoreMetric
//
//  Created by Ege Kaya on 27.11.2025.
//

import Foundation

struct SuspectProcess: Identifiable {
    let id = UUID()
    let pid: String
    let name: String
    let cpu: Double
}

class ProcessInspector {
    
    static func getTopProcesses() -> [SuspectProcess] {
        let task = Process()
        task.launchPath = "/bin/ps"
        task.arguments = ["-Ac", "-o", "pid,%cpu,comm", "-r"]
        
        let pipe = Pipe()
        task.standardOutput = pipe
        
        do {
            try task.run()
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            if let output = String(data: data, encoding: .utf8) {
                return parsePSOutput(output)
            }
        } catch {
            print("Error running ps: \(error)")
        }
        
        return []
    }
    
    private static func parsePSOutput(_ output: String) -> [SuspectProcess] {
        var results: [SuspectProcess] = []
        let lines = output.components(separatedBy: .newlines)

        for line in lines.dropFirst().prefix(3) {
            let parts = line.trimmingCharacters(in: .whitespaces)
                            .components(separatedBy: .whitespaces)
                            .filter { !$0.isEmpty }

            // Format: PID  %CPU  COMMAND
            if parts.count >= 3, let cpu = Double(parts[1]) {
                let pid  = parts[0]
                let name = parts.dropFirst(2).joined(separator: " ")
                results.append(SuspectProcess(pid: pid, name: name, cpu: cpu))
            }
        }
        return results
    }
}
