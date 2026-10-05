// 页码选取状态机：纯前端函数，由 Gradio 的 js= 事件直接调用。
// 12 个形参与 Python 侧事件绑定的组件顺序严格一一对应（顺序敏感，调整组件时需同步修改）：
//   tiers              可用档位名列表（如 ["flash","basic","standard","advanced"]）
//   flashOnlyExtensions 仅支持 flash 档位的扩展名列表
//   maxPages           单次解析页数上限（null 表示不限）
//   file               上传文件（多文件组件传数组，取首文件）
//   position           档位滑块当前位置（tiers 下标）
//   metadata           后端读取的页数元数据 JSON（{"path","page_count","error"}）
//   previous           上一周期状态 JSON（路径、页数、双滑块值、错误）
//   handleAValue/handleBValue  两个页码滑块的当前值
//   inputAValue/inputBValue    两个页码数字输入框的当前值
//   tierSelection      档位偏好 JSON（{"tier","locked"}，locked 表示被 flash-only 文件锁定）
// 返回 12 项更新数组，依次对应：滑块A、滑块B、输入框A、输入框B、摘要HTML、范围文本、
// 状态JSON、档位滑块交互性、档位提示、档位滑块值、档位标签、档位偏好JSON。
(tiers, flashOnlyExtensions, maxPages, file, position, metadata, previous,
 handleAValue, handleBValue, inputAValue, inputBValue, tierSelection) => {
    // 纯前端状态转换：经原生组件事件读写值，不逐次向 Python 发送拖动或输入请求。
    // 内部元数据使用 JSON 文本，避免不同 Gradio 版本的 JSON 组件封装差异。
    const { text, message } = window.__mineruI18n;
    metadata = JSON.parse(metadata || "{}");
    previous = JSON.parse(previous || "{}");
    tierSelection = JSON.parse(tierSelection);
    // 统一构造原生组件的局部更新，未指定属性保持不变。
    const update = (props = {}) => ({ __type__: "update", ...props });
    // 页数读取错误只作为普通文本显示，禁止把异常内容解释为 HTML。
    const escapeHtml = (raw) => String(raw).replace(/[&<>"']/g, (char) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    })[char]);
    // 多文件组件传数组，页码选取以第一个文件为准，与后端元数据/预览/OCR 取首文件保持一致。
    const firstFile = Array.isArray(file) ? file[0] : file;
    const path = (typeof firstFile === "string" ? firstFile : firstFile?.path) || "";
    const flashOnly = flashOnlyExtensions.some((extension) => path.toLowerCase().endsWith(`.${extension}`));
    const flashPosition = tiers.indexOf("flash");
    const flashUnavailable = flashOnly && flashPosition < 0;
    // 只记住未锁定时的用户选择；文件切换和晚到元数据不能把临时 Flash 写回偏好。
    if (!tierSelection.locked) tierSelection.tier = tiers[position];
    const effectivePosition = flashOnly && !flashUnavailable ? flashPosition : tiers.indexOf(tierSelection.tier);
    const selectedTier = flashOnly ? "flash" : tiers[effectivePosition];
    tierSelection.locked = flashOnly;
    const isPdf = path.toLowerCase().endsWith(".pdf");
    const needsRange = isPdf;
    const fileChanged = previous?.path !== path;
    const state = !fileChanged ? { ...previous } : {
        path, page_count: 0, handle_a: 1, handle_b: 1, start_handle: "a",
        metadata_error: "", input_error: "",
    };

    // 只缓存当前文件的元数据；晚到的旧文件响应不能覆盖已经读好的新文件页数。
    let initialized = false;
    if (isPdf && metadata?.path === path && !state.page_count && !state.metadata_error) {
        state.page_count = metadata.page_count || 0;
        state.metadata_error = metadata.error || "";
        state.handle_a = 1;
        state.handle_b = Math.min(state.page_count || 1, maxPages ?? Infinity);
        state.start_handle = "a";
        state.input_error = "";
        initialized = true;
    }

    const count = state.page_count || 0;
    if (count > 0 && !fileChanged && !initialized) {
        // 物理滑块只受文档边界约束；交叉时不交换组件值，保持鼠标和焦点所在的实体。
        const clamp = (value) => Math.min(count, Math.max(1, Math.round(value)));
        // 超限时把另一滑块推到拖动端附近，方向由实际位置而不是起止角色决定。
        const linkedPosition = (active, other) => maxPages !== null && Math.abs(active - other) + 1 > maxPages
            ? active + Math.sign(other - active) * (maxPages - 1)
            : other;
        // 判定本次事件来源：滑块拖动只处理与内部值不一致的实体。
        const sliderAMoved = Number(handleAValue) !== state.handle_a;
        const sliderBMoved = Number(handleBValue) !== state.handle_b;
        if (sliderAMoved || sliderBMoved) {
            if (sliderAMoved) {
                state.handle_a = clamp(handleAValue);
                state.handle_b = linkedPosition(state.handle_a, state.handle_b);
            } else {
                state.handle_b = clamp(handleBValue);
                state.handle_a = linkedPosition(state.handle_b, state.handle_a);
            }
            // 滑块拖动始终合法，清除手动输入遗留的警告。
            state.input_error = "";
        } else {
            // 手动输入只接受 1..页数 的整数；非法时保留用户已键入的内容以便修改，
            // 不回写滑块，也不改变提交范围，仅以警告阻断转换。
            const isIntegerInRange = (value) =>
                typeof value === "number" && Number.isFinite(value) && Number.isInteger(value)
                && value >= 1 && value <= count;
            const inputAChanged = inputAValue !== state.handle_a;
            const inputBChanged = inputBValue !== state.handle_b;
            state.input_error = "";
            if (inputAChanged) {
                if (isIntegerInRange(inputAValue)) {
                    state.handle_a = inputAValue;
                    state.handle_b = linkedPosition(state.handle_a, state.handle_b);
                } else {
                    state.input_error = text("page_out_of_range", { count });
                }
            }
            // 两个输入框同一事件先后校验；若 A 已非法，仍继续校验 B，最终只展示一条警告。
            if (inputBChanged) {
                if (isIntegerInRange(inputBValue)) {
                    state.handle_b = inputBValue;
                    state.handle_a = linkedPosition(state.handle_b, state.handle_a);
                } else if (!state.input_error) {
                    state.input_error = text("page_out_of_range", { count });
                }
            }
        }
    }

    // 只有严格越过才交换角色；重合时沿用上一次角色，避免边界附近的标签抖动。
    if (state.handle_a < state.handle_b) {
        state.start_handle = "a";
    } else if (state.handle_b < state.handle_a) {
        state.start_handle = "b";
    }
    // PDF 页数读取期间保留选页控件的占位，避免切换文件时整列高度跳动。
    const visible = needsRange;
    const interactive = visible && count > 1;
    const start = Math.min(state.handle_a, state.handle_b);
    const end = Math.max(state.handle_a, state.handle_b);
    const selected = end - start + 1;
    const limitText = maxPages === null ? text("page_unlimited") : text("page_limit", { count: maxPages });
    const selectionText = count && !state.input_error
        ? `[${start}-${end}] · ${text("page_count", { count: selected })}`
        : text(state.metadata_error ? "page_read_failed" : "reading_pages");
    const summary = `<div class="mineru-page-values" data-range-visible="${visible}" data-start-handle="${state.start_handle}">`
        + `<span class="mineru-page-selection">${selectionText}</span>`
        + `</div>`
        + `<div class="mineru-page-axis"><span>1</span><span>${limitText}</span><span>${count || "–"}</span></div>`;
    // 手动输入超范围的警告优先于页数元数据错误展示。
    let notice = "";
    if (state.input_error) {
        notice = state.input_error;
    } else if (flashUnavailable) {
        notice = text("flash_unavailable");
    } else if (needsRange && !count) {
        notice = message(state.metadata_error);
    }
    const hasInputError = Boolean(state.input_error);
    // 输入非法时不提交任何页码范围，后端也不会收到越界值。
    const range = count && !hasInputError ? (start === end ? String(start) : `${start}-${end}`) : "";
    // 新版 Gradio 要求非零跨度；单页时禁用控件，实际页数和提交范围仍为 1。
    const maximum = Math.max(2, count);
    const slider = (value, label) => update({ minimum: 1, maximum, value, interactive, label });
    // 数字框始终回填当前合法值，使其与滑块和初始化结果同步；输入非法时上一合法值
    // 仍会下发，但 Gradio 对未失焦的键入不会强制覆盖，用户可继续修改。
    const number = (value, label) => update({ minimum: 1, maximum, value, interactive, label });
    return [
        slider(state.handle_a, state.start_handle === "a" ? text("start_page") : text("end_page")),
        slider(state.handle_b, state.start_handle === "b" ? text("start_page") : text("end_page")),
        number(state.handle_a, state.start_handle === "a" ? text("start_page") : text("end_page")),
        number(state.handle_b, state.start_handle === "b" ? text("start_page") : text("end_page")),
        summary, range, JSON.stringify(state),
        update({ interactive: Boolean(path) && !flashUnavailable && (!needsRange || count > 0) && !hasInputError }),
        update({ value: escapeHtml(notice), visible: Boolean(notice) }),
        // 同一事件同时更新值、标签和页码，程序赋值无需再触发 tier.input。
        update({ value: effectivePosition, interactive: !flashOnly && tiers.length > 1 }),
        text("tier_value", { tier: text(`tier_${selectedTier}`), notice: flashUnavailable ? text("tier_unavailable_suffix") : "" }),
        JSON.stringify(tierSelection),
    ];
}
