# Run log

One entry per task, newest last. Each entry records the date, task id, Slurm job ids, GPU and node, wall time, each acceptance check with pass or fail, the headline numbers and any fix the run needed. Fixes to code are committed separately, and each one is named here with its commit.

## Template

```text
### YYYY-MM-DD, T<n> <job name>
Jobs: <ids>. GPU: <A40 | A100-...>, node <host>. Wall time: <h:mm>.
Acceptance: <check>: pass | fail. <check>: pass | fail.
Numbers: <exact-match rates with 95% intervals, failures, compile times>.
Fixes: <commit and reason, or none>.
Next: <task>.
```
