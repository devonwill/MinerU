// 普通响应携带完整快照；任务身份与序号检查先于任何可见组件更新。
(action, ...args) => {
    const state = window.__mineruConversion ??= { revision: 0, runId: "", sequence: 0, terminal: false, appliedFile: 0 };
    // 使用 Gradio 的空更新保留当前组件，且本脚本的响应不经过生成器差分。
    const skip = () => ({ __type__: "update" });
    const timer = (active) => ({ __type__: "update", active });
    // 容忍已清除的空回执，格式异常也不能破坏当前任务。
    const parse = (value) => {
        try { return JSON.parse(value || "null"); } catch { return null; }
    };
    // randomUUID 仅安全上下文（HTTPS/回环地址）可用；纯 HTTP 局域网访问降级到全上下文可用的 getRandomValues。
    const uuid4 = () => {
        if (crypto.randomUUID) return crypto.randomUUID();
        console.warn("[MinerU WebUI] insecure context, using getRandomValues fallback for run id");
        const b = crypto.getRandomValues(new Uint8Array(16));
        b[6] = (b[6] & 0x0f) | 0x40;
        b[8] = (b[8] & 0x3f) | 0x80;
        return [...b].map((x) => x.toString(16).padStart(2, "0")).join("-");
    };
    // 每个任务最多保留一个预览加载监听器，换文件或重新提交立即回收。
    const stopPreviewLog = () => {
        if (state.previewLoad) document.removeEventListener("load", state.previewLoad, true);
        state.previewLoad = null;
    };

    if (action === "begin") {
        stopPreviewLog();
        state.runId = uuid4().replaceAll("-", "");
        state.revision += 1;
        state.sequence = 0;
        state.terminal = false;
        state.appliedFile = 0;
        window.__mineruStatusPanel?.showPreparing();
        return [JSON.stringify({ run_id: state.runId, revision: state.revision }), timer(true)];
    }
    if (action === "cancel") {
        stopPreviewLog();
        const previous = state.runId ? JSON.stringify({ run_id: state.runId, revision: state.revision }) : "";
        state.runId = "";
        state.terminal = true;
        state.appliedFile = 0;
        return ["", previous, timer(false)];
    }
    if (action === "status") {
        const snapshot = parse(args[0]);
        if (!snapshot || !state.runId || snapshot.run_id !== state.runId || state.terminal
            || !Number.isInteger(snapshot.sequence) || snapshot.sequence <= state.sequence) return [skip(), skip()];
        state.sequence = snapshot.sequence;
        state.terminal = Boolean(snapshot.terminal);
        return [snapshot.html, timer(!state.terminal)];
    }
    if (action === "result") {
        const receipt = parse(args[0]);
        // 返回 22 个可见/文件输出和 Timer，共 23 项；服务端的 artifact State 不经过浏览器写入。
        // 22 = convert_outputs 共 23 项减去 index 6 的 artifact State（含译文预览、译文 Tab 与 5 个译文按钮）。
        // 外层 receipt.change 会再追加 1 个 PDF 查看器输出（共 24 项），校验失败时同样返回 24 个空更新。
        const fileIndex = Number.isInteger(receipt?.file_index) ? receipt.file_index : 1;
        // 队列按文件序号去重：同 runId 下每个文件的回执只应用一次，最后一份才进入终态。
        if (!receipt || !state.runId || receipt.run_id !== state.runId || fileIndex <= state.appliedFile
            || !Array.isArray(receipt.outputs) || receipt.outputs.length !== 22
            || !Number.isInteger(receipt.sequence) || receipt.sequence < state.sequence) {
            return Array.from({ length: 24 }, skip);
        }
        const isTerminal = fileIndex >= (Number.isInteger(receipt.total_files) ? receipt.total_files : 1)
            && Boolean(receipt.terminal);
        state.sequence = receipt.sequence;
        state.appliedFile = fileIndex;
        state.terminal = isTerminal;
        console.info("[MinerU WebUI] result received", JSON.stringify({
            run_id: receipt.run_id, file: `${fileIndex}/${receipt.total_files}`,
            ready_at: receipt.ready_at, received_at: Date.now() / 1000,
        }));
        stopPreviewLog();
        // 在框架写入 iframe 之前捕获 load，避免快速 srcdoc 在下一动画帧前已经加载。
        if (receipt.outputs[1]) {
            state.previewLoad = (event) => {
                if (!event.target?.matches?.("iframe.mineru-rendered-html-frame")) return;
                stopPreviewLog();
                if (state.runId === receipt.run_id) console.info("[MinerU WebUI] preview loaded", JSON.stringify({
                    run_id: receipt.run_id, loaded_at: Date.now() / 1000,
                }));
            };
            document.addEventListener("load", state.previewLoad, true);
        }
        // 中间文件保持轮询活跃，只有最后一份回执停止 Timer。
        return [...receipt.outputs, timer(!isTerminal)];
    }
    return [];
}
