# DeepGEMM Blackwell (SM100) BF16 GEMM 内核逐段讲解

> 对应文件：`deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm.cuh`
>
> 本文是一次"逐行/逐变量"学习的完整记录，面向从未接触过 Blackwell 特性的读者。

---

## 第 0 部分：Blackwell 必备的核心概念

Blackwell (`sm_100`) 相对 Hopper (`sm_90`) 在 GEMM 编程模型上的颠覆性变化：

1. **TMEM（Tensor Memory，张量内存）**
   - 一块全新的、独立的片上内存，专门存放 MMA 的**累加器**（Hopper 时代累加器放寄存器）。
   - 既不是 shared memory 也不是寄存器，是第三种存储；需要**显式分配 / 释放**。
   - 形状固定 **128 行 × 若干列**。

2. **`tcgen05` —— 第五代 Tensor Core 指令**
   - MMA 变成"**单线程发射、异步执行**"：一个线程发一条 `tcgen05.mma`，硬件后台算，算完用 barrier 通知。

3. **UMMA**
   - Blackwell 新 MMA 抽象（CUTLASS `cute::UMMA`）。直接从 **shared memory 读 A、B**，结果写进 **TMEM**。
   - 用 **descriptor（描述符）** 告诉硬件 A/B 在 smem 的位置、布局、swizzle。

4. **TMA（Tensor Memory Accelerator）**（Hopper 已有）
   - 异步拷贝引擎：一条指令搬一整块 tile 从 global → shared memory。
   - 用 **TmaDescriptor** 描述 global 张量；完成通过 **mbarrier** 通知。

5. **Cluster + Multicast（2-CTA / 2-SM 协作）**
   - 2 个 CTA 组成 cluster，可共享加载（multicast，一次读广播给两块）。
   - MMA 也能 2-SM 协作（`2x1SM`），两个 SM 的 Tensor Core 合算一个更大的 tile。

6. **Warp Specialization（内核组织方式）**
   - 不是所有线程干一样的活，而是**按 warp 分工**：

   | Warp | 角色 | 职责 |
   |------|------|------|
   | warp 0 | TMA load warp | 把 A/B 从显存搬到 shared memory |
   | warp 1 | MMA issue warp | 发射 UMMA 让 Tensor Core 计算 |
   | warp 2 | 分配 TMEM | |
   | 后续一批 warp | Epilogue warps | 把 TMEM 结果搬到 smem 再写回显存 |

   数据流：
   ```
   显存 --(TMA,warp0)--> smem --(UMMA,warp1)--> TMEM --(epilogue)--> smem --(TMA store)--> 显存
   ```

---

## 第 1 部分：头文件 + 函数签名（第 1–37 行）

- `#pragma once` 防重复包含；`clang diagnostic push/ignored` 临时关掉 unknown-attributes 警告（文件末尾 `pop` 恢复）。
- 头文件对应各模块：barrier、cluster 同步、调度器、TMA 拷贝、epilogue 写回（普通/swap-AB）、UMMA 描述符、tcgen05 PTX。

### 模板参数

| 参数 | 含义 |
|------|------|
| `kMajorA/kMajorB` | A、B 的主序：`Major::K` 或 `Major::MN` |
| `SHAPE_M/N/K` | 编译期形状（为 0 则用运行时 `shape_*`） |
| `BLOCK_M/N/BLOCK_K_` | tile 大小（`BLOCK_K_` 后面可能"合并 stage"重算） |
| `kNumGroups` | 分组 GEMM 组数，普通为 1 |
| `kSwizzleA/B/CDMode` | shared memory swizzle 模式（字节数），防 bank conflict |
| `kNumStages_` | 流水线级数（隐藏延迟） |
| `kNumNonEpilogueThreads` | 非尾声线程数（TMA+MMA+分配 TMEM） |
| `kNumEpilogueThreads` | 尾声线程数 |
| `kNumMulticast` | multicast 数（1 或 2） |
| `kIsMulticastOnA` | multicast 作用在 A 还是 B |
| `kNumSMs` | 使用的 SM 数（persistent 调度） |
| `kSwapAB` | 是否交换 A/B |
| `kGemmType` | Normal / MGroupedMasked / KGroupedContiguous / Batched 等 |
| `kWithAccumulation` | 结果是否累加到已有 C |
| `cd_dtype_t` | 输出类型：`float` 或 `bfloat16_t` |
| `kTensorCoreUtilControl` | Tensor Core 利用率控制（<100 表示限速防降频） |

### 函数签名
- `__launch_bounds__(总线程数, 1)`：每 block 线程数 + 每 SM 至少 1 个 block。
- 参数：`grouped_layout`（分组信息）、运行时 `shape_*`、三个 `__grid_constant__ TmaDescriptor`（A/B/CD）。

---

## 第 2 部分：编译期常量推导（第 38–95 行）

- **架构守卫**（38）：只有 `__CUDA_ARCH__ >= 1000`（Blackwell）才编译内核体。
- **Stage 合并优化**（39–48）：当 stage≥8、Normal、A/B 都 K-major 时，把多个小 stage 合成大 stage（`BLOCK_K` 变大、stage 变少），减少 `umma_arrive` 开销。保留至少 8 个 stage。
- **类型别名**（50–54）：`Barrier = ClusterTransactionBarrier`（能追踪到达字节数）；`Allocator` 按 multicast 选 1-SM/2-SM 版。
- **UMMA 形状**（56–66）：
  - `LAYOUT_AD_M=128`（TMEM 128 行）；`UMMA_M = 128 * kNumMulticast`（2-SM 时 256）；`UMMA_N = kSwapAB ? BLOCK_M : BLOCK_N`；`UMMA_K=16`。
  - `LOAD_BLOCK_M/N`：每个 CTA 实际加载大小（按 multicast 是否作用在该操作数分摊）。
- **Epilogue 配置**（68–77）：`kNumEpilogueStages=2`、`kNumTMAStoreStages=2`；`STORE_BLOCK_M/N`、`kNumUMMAStoreThreads`。
- **Shared memory 尺寸**（79–90）：`SMEM_CD/A/B_SIZE*`，都要 1024 字节对齐；UMMA padding 校验。
- **TMEM 列数**（92–95）：`kNumAccumTmemCols = kNumEpilogueStages * UMMA_N`，向上对齐到合法列数（32–512）。

---

## 第 3 部分：运行前准备（第 97–142 行）

- **cluster 同步**（97–98）：2-SM 才需要（下面要 2-CTA 联合分配 TMEM）。
- **身份**（101–103）：`is_leader_cta`、`warp_idx`（分角色）、`lane_idx`。
- **预取 TMA 描述符**（105–110）：warp 0 提前把三个描述符取进 cache。
- **覆盖形状**（112–115）：编译期非 0 就用编译期值。
- **shared memory 手动布局**（117–129）：
  ```
  [ CD 区 ][ A×kNumStages ][ B×kNumStages ][ barriers ][ tmem 指针 ]
  ```
  `PatternVisitor` 是"给 stage 序号返回该缓冲指针"的下标访问器。
- **barrier 布局**（131–137）：
  | 名称 | 数量 | 作用 |
  |------|------|------|
  | `full_barriers` | kNumStages | A/B 到齐（TMA arrive，MMA wait） |
  | `empty_barriers` | kNumStages | A/B 缓冲腾空（MMA arrive，TMA wait） |
  | `tmem_full_barriers` | kNumEpilogueStages | 累加器算完（MMA arrive，epilogue wait） |
  | `tmem_empty_barriers` | kNumEpilogueStages | 累加器搬空（epilogue arrive，MMA wait） |
  | `tensor_core_full_barrier` | 1 | 限速用 |
- **TMEM 指针占位**（139–141）：warp 2 分配 TMEM 后把基址写进这个 smem 位置，其他 warp 读取。

---

## 第 4 部分：初始化 + 调度器（第 143–186 行）

- **初始化 barrier**（143–163）：warp 1 的一个线程 `init(N)` 设定每个 barrier 的"期望 arrive 次数"；`fence_barrier_init()` 让初始化对异步代理可见。
  - `full_barriers->init(kNumMulticast)`；`empty_barriers->init(1)`；
  - `tmem_full_barriers->init(1)`；`tmem_empty_barriers->init(kNumMulticast * kNumUMMAStoreThreads)`（epilogue 很多线程各 arrive 一次）。
- **分配 TMEM**（164–167）：warp 2 并行执行 `Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem)`。
- **全员同步**（168）。
- **等前置 kernel**（170–171）：`cudaGridDependencySynchronize()`（PDL，程序化依赖启动，重叠上个 kernel 收尾）。
- **构造调度器**（173–176）：persistent kernel，一个 block 循环领多个 tile。
- **流水线状态 + 推进函数**（178–186）：
  - `stage_idx`（0..kNumStages-1 循环）、`phase`（0/1 翻转，用于 mbarrier 可重用的相位匹配）。
  - `advance_pipeline`：k+1、stage 前进、绕回 0 时翻 phase。

---

## 第 5 部分：TMA load warp（warp 0，第 188–249 行）

生产者逻辑（每 tile 每 K-block）：
1. **等** `empty_barriers[stage_idx]->wait(phase^1)`：确认 MMA 用完这块缓冲。
2. **算**全局偏移 `m_idx/n_idx/k_a_idx/k_b_idx`；2-SM 时加 `block_rank * load 尺寸` 偏移。
3. **发** TMA：四个 `if constexpr` 按主序选 K-major / MN-major 的布局，把 A/B 搬进 `smem_a/b[stage_idx]`，完成后自动 arrive `full_barriers[stage_idx]`。
4. **声明字节** `arrive_and_expect_tx(kNumArrivalBytes * kNumMulticast)`（仅 leader；非 leader `arrive(0)`）。

---

## 第 6 部分：MMA issue warp（warp 1，leader CTA，第 250–368 行）

双身份：A/B 的消费者 + TMEM 累加器的生产者。

- **指令描述符 `instr_desc`**（254–257）：MMA 形状/类型/主序（swap 时对调 A/B 主序）。
- **SMEM 描述符 `a_desc/b_desc`**（262–265）：A/B 在 smem 的地址/布局；用 `a_desc_lo/b_desc_lo` + `__shfl_sync` 做"stage→地址"查表（第 `lane` 号线程存第 `lane` 个 stage 的地址）。
- **主循环**：
  1. `tmem_empty_barriers[accum_stage_idx]->wait`：等 epilogue 腾空累加器。
  2. `umma_arrive` / `empty_barrier_arrive` lambda：封装 `tcgen05.commit`（见下）。
  3. K 循环，每 K-block：
     - `full_barriers[stage_idx]->wait(phase)`：等 A/B 到齐。
     - `__shfl_sync` 取当前 stage 的描述符地址。
     - `elect_one_sync()` 单线程发 UMMA：内层按 `UMMA_K=16` 切片，`advance_umma_desc_lo` 推进地址，`mma_t::fma(A, B, accum_stage_idx*UMMA_N, 累加标志, desc)`。
       - 累加标志 `k_block_idx>0 or k>0`：首片写入、后续累加，实现 `C=Σ_k A_k·B_k`。
     - `empty_barrier_arrive(是否最后K块)`：通知 `empty_barriers`（A/B 用完），最后一块再通知 `tmem_full_barriers`（累加器算完）。
     - 可选限速：`tensor_core_full_barrier` 等 UMMA 算完后忙等 `kNumDummyCycles`。
- **收尾**（363–368）：2-SM 时再等一次最后的 `tmem_empty_barriers`，保证安全析构。

### 关于 `tcgen05.commit`（常见疑问）
- `commit` 藏在 `cutlass::arch::umma_arrive` 里，本质是 `tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64`，一条指令完成"提交异步 UMMA 归组 + 完成后 arrive mbarrier"。
- **wait 不在 MMA warp**：由消费者做——TMA warp 等 `empty_barriers`、epilogue 等 `tmem_full_barriers`。这正是流水线不阻塞的关键。
- 唯一 MMA warp 自己 wait 的地方是限速逻辑。

---

## 第 7 部分：Epilogue warps + 收尾（第 369–433 行）

- **认领角色**（369–376）：warp_idx 落在 `[kNumNonEpilogueThreads/32, +kNumUMMAStoreThreads/32)` 区间；断言 TMEM 基址为 0。
- **主循环**（382–418）：
  1. `tmem_full_barriers[accum_stage_idx]->wait(accum_phase_idx)`：等 MMA 算完累加器（`accum_stage_idx/phase` 算法与 MMA warp 一致）。
  2. 算输出坐标 `base_m_idx/base_n_idx`、`tmem_base_addr = accum_stage_idx * UMMA_N`。
  3. 调 `sm100_store_cd`（非 swap）或 `sm100_store_cd_swap_ab`（swap）：`TMEM→寄存器→(STSM)smem→(TMA store)显存`，用 `kNumTMAStoreStages` 多缓冲流水。
  4. store 内部搬完 `tmem_empty_barrier->arrive`：通知 MMA 累加器可复用。
- **收尾**：全员同步（421–423）→ warp 0 `Allocator().free`（425–427）。
- **兜底**（429–432）：非 Blackwell 架构直接断言报错。

---

## 全局总结

```
        warp 0: TMA load            warp 1: MMA(leader)              epilogue warps
显存A/B ─TMA─▶ smem[A/B]  ─UMMA─▶ TMEM 累加器  ─STSM─▶ smem[CD] ─TMA store─▶ 显存C/D
         (kNumStages)              (kNumEpilogueStages)        (kNumTMAStoreStages)
        full/empty barriers      tmem_full/empty barriers    tma_store_wait 流水
```

三条流水线的多缓冲：
- `kNumStages`：TMA↔MMA，A/B 输入（shared memory）。
- `kNumEpilogueStages`：MMA↔epilogue，累加器（TMEM）。
- `kNumTMAStoreStages`：epilogue 写回（shared memory）。

核心套路：warp specialization + persistent 调度 + 2-SM cluster/multicast + TMEM 显式管理 + tcgen05 异步 UMMA + TMA 异步搬运 + mbarrier phase 流水线。

---

## 附：几个关键问答

### Q1. `UMMA_M = 128*kNumMulticast` 已经用了 2 个 SM 在 M 维，为什么 `LOAD_BLOCK_M` 还要看 `kIsMulticastOnA`？
两个正交的层面：
- `UMMA_M`（**计算层**）：`2x1SM` atom 硬件规定两个 SM 沿累加器 M 轴叠加，固定，无关 A/B。
- `kIsMulticastOnA`（**搬运层**）：TMA 加载时这对 CTA 沿 A 还是 B 分摊搬运、哪个操作数被复用/共享——带宽/L2 调优选择，还会决定调度器 swizzle 分组方向（group on N vs M）。
  - `true`：A 按 M 切两半各搬一半，B 两 CTA 各搬完整。
  - `false`：B 按 N 切两半，A 两 CTA 各搬完整。

### Q2. swap-AB 时为什么 `STORE_BLOCK_M` 固定是 16？
swap-AB 时结果在 TMEM 里是**转置**存放（TMEM 128 行=输出 N；列=输出 M）。写回必须用硬件**转置版 store-matrix** `stmatrix.trans`（`SM90_U32x4_STSM_T`）。该指令 BF16 处理固定 **16×16** 原子块（32 线程 × 4 uint32 × 2 bf16 = 256 = 16×16），转置后 M 方向宽度正好 16，故 `STORE_BLOCK_M=16`。

### Q3. `kNumTMAStoreStages` 是"写几次"吗？
不是。它是**写回流水线的 shared memory 缓冲区个数**（多缓冲深度），让 STSM(TMEM→smem) 与 TMA store(smem→显存) 重叠。
"分几次写" = `BLOCK_M/STORE_BLOCK_M × BLOCK_N/STORE_BLOCK_N`。例：`BLOCK_N=64, STORE_BLOCK_N=32` → N 方向分 2 次。

### Q4. `kNumEpilogueStages` vs `kNumTMAStoreStages`？
- `kNumEpilogueStages`：**TMEM** 累加器缓冲个数，MMA(写)↔epilogue(读) 重叠。
- `kNumTMAStoreStages`：**shared memory** 写回缓冲个数，STSM(写)↔TMA store(读) 重叠。

### Q5. `arrive_and_expect_tx` 为什么放在 `tma::copy` 之后？
前后都对。`arrive_and_expect_tx` 里的 `arrive` 是 barrier 翻转的"总闸"——调用它之前 arrival count 未满足，barrier 绝不会翻转，故 expect 与 TMA 完成（complete_tx）谁先谁后无所谓（对同一 tx 计数器加/减，净归零才唤醒）。放"之后"是为了让异步 TMA 尽早启动。
