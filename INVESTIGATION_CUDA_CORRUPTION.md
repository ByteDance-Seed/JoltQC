# CUDA State Corruption Investigation

## Issue Summary
Running `test_jk_2d.py` followed by `test_linalg_helper.py` causes CUDA illegal memory access errors in subsequent tests.

## Root Cause Analysis

### Symptoms
1. **Error Type**: `CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered`
2. **Location**: Occurs in `cupy.cuda.function.Module.__dealloc__` during module unload
3. **Timing**: Happens during Python garbage collection after test_jk_2d completes
4. **Impact**: Corrupts CUDA context, causing all subsequent CUDA operations to fail

### Investigation Findings

#### 1. Not Caused by `free_all_blocks()`
- Initially suspected aggressive memory cleanup in tearDown
- Removing `free_all_blocks()` did NOT fix the issue
- Error persists even with natural memory management

#### 2. Tests Pass Individually
- `test_jk_2d.py` alone: ✅ All 8 tests pass
- `test_linalg_helper.py` alone: ✅ All 4 tests pass
- Together in sequence: ❌ CUDA corruption

#### 3. Cumulative Effect
- Running 1-2 tests from test_jk_2d: Works fine
- Running all 8 tests from test_jk_2d: Causes corruption
- Suggests cumulative CUDA resource exhaustion or memory corruption

### Possible Causes

#### A. Kernel Memory Corruption (MOST LIKELY)
The 2D VJ/VK kernels may have out-of-bounds memory accesses:
- Buffer overruns in shared memory
- Incorrect array indexing
- Thread synchronization issues
- Improper handling of BASIS_STRIDE in packed basis access

**Evidence:**
- Errors occur during module cleanup, suggesting corrupted CUDA state
- Only happens after multiple kernel executions
- Standalone kernel runs work, but cumulative runs fail

#### B. CuPy Module Management Issue
CuPy's RawModule may not properly clean up compiled kernels:
- Modules hold references to CUDA resources
- Garbage collection happens at unpredictable times
- CUDA driver state becomes inconsistent

#### C. CUDA Driver/CuPy Compatibility
- Potential incompatibility between CUDA 12.4 and CuPy version
- Driver-level resource leaks

## Recommended Actions

### Immediate (High Priority)
1. **Add CUDA Error Checking to Kernels**
   - Use `cuda-memcheck` to detect out-of-bounds accesses
   - Add assertions in CUDA kernels for array bounds
   - Verify BASIS_STRIDE calculations are correct

2. **Validate Kernel Memory Accesses**
   ```bash
   compute-sanitizer python -m pytest jqc/backend/tests/test_jk_2d.py
   ```

3. **Check Shared Memory Usage**
   - Verify shared memory allocations don't exceed limits
   - Check for bank conflicts or race conditions

### Medium Priority
4. **Reduce Test Scope**
   - Mark test_jk_2d as requiring isolation
   - Document that these tests must run separately
   - Update pytest configuration to isolate GPU tests

5. **Add Explicit Module Cleanup**
   - Store module references and explicitly delete them
   - Force garbage collection between test classes
   - Reset CUDA context if possible

### Long-term
6. **Refactor Test Strategy**
   - Split heavy compilation tests into separate test suite
   - Use mocking for signature tests (don't actually execute kernels)
   - Create minimal kernels for testing (s-s-s-s only)

## Current Workaround

**Run tests in isolation:**
```bash
pytest jqc/backend/tests/test_jk_2d.py
pytest jqc/backend/tests/test_linalg_helper.py
pytest jqc/backend/tests/test_basis_layout.py
```

**Do NOT run:**
```bash
pytest jqc/backend/tests/  # This will fail
```

## Test Results

### Individual Test Runs ✅
- `test_jk_2d.py`: 8/8 passed
- `test_linalg_helper.py`: 4/4 passed
- `test_basis_layout.py`: 12/12 passed

### Combined Test Run ❌
- `test_jk_2d.py + test_linalg_helper.py`: 8 passed, 4 failed
- CUDA context corruption after test_jk_2d
- Illegal memory access errors during module cleanup

## Next Steps

1. Run CUDA memory checker on the kernels
2. Review BASIS_STRIDE indexing in 2d_vj.cu and 2d_vk.cu
3. Check if issue exists on refactor_basis branch (before 2-fold-sym merge)
4. Consider adding CUDA error checks after each kernel launch in tests
