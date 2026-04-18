import Foundation
import Darwin.Mach

/// Small cross-backend metrics helpers exposed to app code.
public enum KittenMetrics {

    /// Resident set size of the current process, in bytes.
    /// Uses Mach's `task_vm_info` (more accurate on iOS than mach_task_basic_info).
    public static func residentBytes() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info>.size / MemoryLayout<integer_t>.size)
        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) { ptr in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { iptr in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), iptr, &count)
            }
        }
        return kerr == KERN_SUCCESS ? info.phys_footprint : 0
    }

    public static func residentMB() -> Double {
        Double(residentBytes()) / 1024.0 / 1024.0
    }

    public static func formatMB(_ bytes: UInt64) -> String {
        let mb = Double(bytes) / 1024.0 / 1024.0
        return String(format: "%.0f MB", mb)
    }
}
