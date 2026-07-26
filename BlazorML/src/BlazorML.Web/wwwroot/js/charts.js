/* ============================================================================
   Evaluation charts.

   Four forms, each picked for the job its data does:
     ROC          → line. A trade-off traced across thresholds.
     Confusion    → heatmap. Magnitude in a matrix, one hue light→dark.
     Residuals    → scatter. Two continuous measures against each other.
     Importance   → horizontal bars. Ranked magnitude with long labels, and a
                    diverging scale because permutation importance can go
                    negative — that is a real finding, not noise.

   Colour comes from the CSS custom properties so both themes are handled by the
   same code path, and the dark steps are the validated ones rather than an
   automatic flip. Every chart ships a hover layer and pairs with a table view
   rendered by the Blazor component, so nothing here is colour-alone.
   ========================================================================= */

(function () {
    const AXIS_PAD = { top: 12, right: 14, bottom: 34, left: 46 };

    function token(name, fallback) {
        const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
        return value || fallback;
    }

    function clear(host) {
        d3.select(host).selectAll('*').remove();
    }

    /** Shared shell: sized SVG, recessive grid, axes in muted ink. */
    function frame(host, { xLabel, yLabel, xDomain, yDomain, height = 240 }) {
        clear(host);

        const width = Math.max(240, host.clientWidth || 320);
        const inner = {
            w: width - AXIS_PAD.left - AXIS_PAD.right,
            h: height - AXIS_PAD.top - AXIS_PAD.bottom
        };

        const svg = d3.select(host).append('svg')
            .attr('width', '100%')
            .attr('viewBox', `0 0 ${width} ${height}`)
            .attr('role', 'img');

        const g = svg.append('g').attr('transform', `translate(${AXIS_PAD.left},${AXIS_PAD.top})`);

        const x = d3.scaleLinear().domain(xDomain).nice().range([0, inner.w]);
        const y = d3.scaleLinear().domain(yDomain).nice().range([inner.h, 0]);

        const grid = token('--chart-grid', '#e2e2dd');
        const axis = token('--chart-axis', '#8b93a1');

        g.append('g')
            .attr('class', 'c-grid')
            .selectAll('line')
            .data(y.ticks(5))
            .join('line')
            .attr('x1', 0).attr('x2', inner.w)
            .attr('y1', d => y(d)).attr('y2', d => y(d))
            .attr('stroke', grid)
            .attr('stroke-width', 1);

        const xAxis = g.append('g')
            .attr('transform', `translate(0,${inner.h})`)
            .call(d3.axisBottom(x).ticks(5).tickSize(4));

        const yAxis = g.append('g').call(d3.axisLeft(y).ticks(5).tickSize(4));

        for (const a of [xAxis, yAxis]) {
            a.selectAll('path,line').attr('stroke', axis);
            a.selectAll('text').attr('fill', axis).style('font-size', '10px').style('font-family', 'var(--font-mono)');
        }

        svg.append('text')
            .attr('x', AXIS_PAD.left + inner.w / 2).attr('y', height - 4)
            .attr('text-anchor', 'middle').attr('fill', axis)
            .style('font-size', '10px').text(xLabel);

        svg.append('text')
            .attr('transform', `rotate(-90)`)
            .attr('x', -(AXIS_PAD.top + inner.h / 2)).attr('y', 12)
            .attr('text-anchor', 'middle').attr('fill', axis)
            .style('font-size', '10px').text(yLabel);

        return { svg, g, x, y, inner };
    }

    /** One tooltip element per host, reused across redraws. */
    function tooltip(host) {
        let tip = host.querySelector('.c-tip');

        if (!tip) {
            tip = document.createElement('div');
            tip.className = 'c-tip';
            tip.hidden = true;
            host.appendChild(tip);
        }

        return {
            show(html, x, y) {
                tip.innerHTML = html;
                tip.hidden = false;
                tip.style.left = `${x}px`;
                tip.style.top = `${y}px`;
            },
            hide() { tip.hidden = true; }
        };
    }

    window.blazormlCharts = {
        /** ROC curve with the no-skill diagonal as a reference line. */
        roc(host, points, auc) {
            if (!host || !points || points.length < 2) return;

            const { g, x, y, inner } = frame(host, {
                xLabel: 'False positive rate', yLabel: 'True positive rate',
                xDomain: [0, 1], yDomain: [0, 1]
            });

            // The diagonal is what a coin flip looks like. Without it a ROC curve
            // has no scale of "good".
            g.append('line')
                .attr('x1', x(0)).attr('y1', y(0)).attr('x2', x(1)).attr('y2', y(1))
                .attr('stroke', token('--chart-reference', '#b4b8b0'))
                .attr('stroke-width', 2)
                .attr('stroke-dasharray', '5 5');

            const pen = token('--pen-lime', '#5c9a1b');
            const line = d3.line().x(d => x(d.falsePositiveRate)).y(d => y(d.truePositiveRate));

            g.append('path')
                .datum(points)
                .attr('fill', 'none')
                .attr('stroke', pen)
                .attr('stroke-width', 2)
                .attr('stroke-linejoin', 'round')
                .attr('d', line);

            if (typeof auc === 'number' && !Number.isNaN(auc)) {
                g.append('text')
                    .attr('x', inner.w - 6).attr('y', inner.h - 8)
                    .attr('text-anchor', 'end')
                    .attr('fill', token('--ink', '#14161a'))
                    .style('font-size', '12px').style('font-weight', '600')
                    .style('font-family', 'var(--font-mono)')
                    .text(`AUC ${auc.toFixed(3)}`);
            }

            // Crosshair: snap to the nearest point by false-positive rate.
            const tip = tooltip(host);
            const marker = g.append('circle')
                .attr('r', 5).attr('fill', pen)
                .attr('stroke', token('--surface', '#fff')).attr('stroke-width', 2)
                .style('display', 'none');

            g.append('rect')
                .attr('width', inner.w).attr('height', inner.h)
                .attr('fill', 'transparent')
                .on('pointermove', event => {
                    const [px] = d3.pointer(event);
                    const target = x.invert(px);
                    const hit = points.reduce((best, p) =>
                        Math.abs(p.falsePositiveRate - target) < Math.abs(best.falsePositiveRate - target) ? p : best);

                    marker.attr('cx', x(hit.falsePositiveRate)).attr('cy', y(hit.truePositiveRate))
                        .style('display', null);

                    tip.show(
                        `<b>Ambang ${hit.threshold.toFixed(3)}</b><br>` +
                        `TPR ${hit.truePositiveRate.toFixed(3)}<br>FPR ${hit.falsePositiveRate.toFixed(3)}`,
                        x(hit.falsePositiveRate) + AXIS_PAD.left + 12,
                        y(hit.truePositiveRate) + AXIS_PAD.top);
                })
                .on('pointerleave', () => { marker.style('display', 'none'); tip.hide(); });
        },

        /** Confusion matrix as a heatmap: one hue, light to dark, counts on the cells. */
        confusion(host, labels, matrix) {
            if (!host || !labels || !matrix || matrix.length === 0) return;

            clear(host);

            const n = labels.length;
            const cell = Math.max(34, Math.min(64, Math.floor((host.clientWidth || 320) / (n + 1.6))));
            const pad = { left: 78, top: 24 };
            const width = pad.left + cell * n + 10;
            const height = pad.top + cell * n + 34;

            const svg = d3.select(host).append('svg')
                .attr('width', '100%')
                .attr('viewBox', `0 0 ${width} ${height}`)
                .attr('role', 'img');

            const max = d3.max(matrix.flat()) || 1;

            // Sequential ramp read from the design tokens, so dark mode uses its own
            // validated steps rather than an inverted light ramp.
            const ramp = [0, 1, 2, 3, 4, 5].map(i => token(`--seq-${i}`, '#cfe0fa'));
            const colour = d3.scaleQuantize().domain([0, max]).range(ramp);

            const ink = token('--ink', '#14161a');
            const axis = token('--chart-axis', '#8b93a1');
            const surface = token('--surface', '#fff');
            const tip = tooltip(host);

            for (let r = 0; r < n; r++) {
                svg.append('text')
                    .attr('x', pad.left - 8).attr('y', pad.top + r * cell + cell / 2 + 4)
                    .attr('text-anchor', 'end').attr('fill', axis)
                    .style('font-size', '11px').text(labels[r]);

                for (let c = 0; c < n; c++) {
                    const value = matrix[r][c] ?? 0;
                    const strong = value > max * 0.55;

                    svg.append('rect')
                        // 2px gap between fills so adjacent cells never bleed together.
                        .attr('x', pad.left + c * cell + 1).attr('y', pad.top + r * cell + 1)
                        .attr('width', cell - 2).attr('height', cell - 2)
                        .attr('rx', 4)
                        .attr('fill', colour(value))
                        .attr('stroke', r === c ? ink : 'none')
                        .attr('stroke-width', r === c ? 2 : 0)
                        .style('cursor', 'default')
                        .on('pointerenter', function (event) {
                            tip.show(
                                `Sebenarnya <b>${labels[r]}</b><br>Diprediksi <b>${labels[c]}</b><br>${value} baris`,
                                pad.left + c * cell + cell, pad.top + r * cell);
                        })
                        .on('pointerleave', () => tip.hide());

                    svg.append('text')
                        .attr('x', pad.left + c * cell + cell / 2)
                        .attr('y', pad.top + r * cell + cell / 2 + 4)
                        .attr('text-anchor', 'middle')
                        .attr('fill', strong ? surface : ink)
                        .style('font-size', '11px').style('font-weight', '600')
                        .style('font-family', 'var(--font-mono)')
                        .style('pointer-events', 'none')
                        .text(value);
                }
            }

            for (let c = 0; c < n; c++) {
                svg.append('text')
                    .attr('x', pad.left + c * cell + cell / 2).attr('y', height - 14)
                    .attr('text-anchor', 'middle').attr('fill', axis)
                    .style('font-size', '11px').text(labels[c]);
            }

            svg.append('text')
                .attr('x', 4).attr('y', 14)
                .attr('fill', axis).style('font-size', '10px')
                .text('baris = sebenarnya · kolom = prediksi');
        },

        /** Actual vs predicted, with the perfect-prediction line as reference. */
        residuals(host, points) {
            if (!host || !points || points.length === 0) return;

            const values = points.flatMap(p => [p.actual, p.predicted]);
            const domain = [d3.min(values), d3.max(values)];

            const { g, x, y, inner } = frame(host, {
                xLabel: 'Nilai sebenarnya', yLabel: 'Prediksi',
                xDomain: domain, yDomain: domain
            });

            g.append('line')
                .attr('x1', x(domain[0])).attr('y1', y(domain[0]))
                .attr('x2', x(domain[1])).attr('y2', y(domain[1]))
                .attr('stroke', token('--chart-reference', '#b4b8b0'))
                .attr('stroke-width', 2)
                .attr('stroke-dasharray', '5 5');

            const pen = token('--pen-teal', '#04938c');
            const tip = tooltip(host);

            g.selectAll('circle')
                .data(points)
                .join('circle')
                .attr('cx', d => x(d.actual)).attr('cy', d => y(d.predicted))
                .attr('r', 4)
                .attr('fill', pen)
                .attr('fill-opacity', 0.55)
                // A surface ring keeps overlapping dots readable as separate marks.
                .attr('stroke', token('--surface', '#fff'))
                .attr('stroke-width', 1.5)
                .on('pointerenter', (event, d) => tip.show(
                    `Sebenarnya <b>${d.actual.toFixed(2)}</b><br>Prediksi <b>${d.predicted.toFixed(2)}</b><br>` +
                    `Selisih ${(d.predicted - d.actual).toFixed(2)}`,
                    x(d.actual) + AXIS_PAD.left + 10, y(d.predicted) + AXIS_PAD.top))
                .on('pointerleave', () => tip.hide());

            g.append('text')
                .attr('x', inner.w - 6).attr('y', 12)
                .attr('text-anchor', 'end').attr('fill', token('--chart-axis', '#8b93a1'))
                .style('font-size', '10px')
                .text('garis putus = prediksi sempurna');
        },

        /**
         * Feature importance. Diverging on purpose: a negative score means the model
         * scored better with that column shuffled, which is worth seeing rather than
         * clamping to zero.
         */
        importance(host, weights) {
            if (!host || !weights || weights.length === 0) return;

            clear(host);

            const top = weights.slice(0, 12);
            const rowH = 26;
            const pad = { left: 128, right: 52, top: 8 };
            const width = Math.max(280, host.clientWidth || 320);
            const height = pad.top + top.length * rowH + 8;
            const plotW = width - pad.left - pad.right;

            const svg = d3.select(host).append('svg')
                .attr('width', '100%')
                .attr('viewBox', `0 0 ${width} ${height}`)
                .attr('role', 'img');

            const extent = d3.max(top, d => Math.abs(d.weight)) || 1;
            const x = d3.scaleLinear().domain([-extent, extent]).range([0, plotW]);
            const zero = pad.left + x(0);

            const positive = token('--pen-teal', '#04938c');
            const negative = token('--pen-rose', '#d6336c');
            const ink = token('--ink', '#14161a');
            const axis = token('--chart-axis', '#8b93a1');

            top.forEach((d, i) => {
                const yTop = pad.top + i * rowH;
                const w = Math.abs(x(d.weight) - x(0));

                svg.append('text')
                    .attr('x', pad.left - 8).attr('y', yTop + rowH / 2 + 4)
                    .attr('text-anchor', 'end').attr('fill', ink)
                    .style('font-size', '11px')
                    .text(d.feature.length > 18 ? d.feature.slice(0, 17) + '…' : d.feature)
                    .append('title').text(d.feature);

                svg.append('rect')
                    .attr('x', d.weight >= 0 ? zero : zero - w)
                    .attr('y', yTop + 4)
                    .attr('width', Math.max(2, w))
                    .attr('height', rowH - 10)
                    // Rounded ends on the data end only; the baseline end stays square.
                    .attr('rx', 4)
                    .attr('fill', d.weight >= 0 ? positive : negative);

                svg.append('text')
                    .attr('x', width - pad.right + 6).attr('y', yTop + rowH / 2 + 4)
                    .attr('fill', axis)
                    .style('font-size', '10px').style('font-family', 'var(--font-mono)')
                    .text(d.weight.toFixed(4));
            });

            svg.append('line')
                .attr('x1', zero).attr('x2', zero)
                .attr('y1', pad.top).attr('y2', height - 8)
                .attr('stroke', axis).attr('stroke-width', 1);
        },

        /**
         * One metric across successive runs. A bar chart because the job is comparing magnitudes
         * between discrete things, and a single series so there is no palette to get wrong — the
         * best run is picked out by weight, not by a second hue.
         */
        runComparison(host, points, higherIsBetter) {
            if (!host || !points || points.length === 0) return;

            clear(host);

            const rowH = 30;
            const pad = { left: 96, right: 62, top: 8 };
            const width = Math.max(300, host.clientWidth || 360);
            const height = pad.top + points.length * rowH + 8;
            const plotW = width - pad.left - pad.right;

            const svg = d3.select(host).append('svg')
                .attr('width', '100%')
                .attr('viewBox', `0 0 ${width} ${height}`)
                .attr('role', 'img');

            const values = points.map(p => p.value).filter(v => Number.isFinite(v));
            if (!values.length) return;

            // Bars start at zero unless the values do not span it — truncating a baseline
            // exaggerates small differences, which is the classic way to mislead with a bar chart.
            const min = Math.min(0, d3.min(values));
            const max = Math.max(0, d3.max(values));
            const x = d3.scaleLinear().domain([min, max]).nice().range([0, plotW]);

            const best = higherIsBetter ? d3.max(values) : d3.min(values);
            const pen = token('--pen-lime', '#5c9a1b');
            const ink = token('--ink', '#14161a');
            const axis = token('--chart-axis', '#8b93a1');
            const tip = tooltip(host);

            points.forEach((p, i) => {
                const y = pad.top + i * rowH;
                const isBest = Number.isFinite(p.value) && Math.abs(p.value - best) < 1e-12;

                svg.append('text')
                    .attr('x', pad.left - 8).attr('y', y + rowH / 2 + 4)
                    .attr('text-anchor', 'end').attr('fill', axis)
                    .style('font-size', '10px').style('font-family', 'var(--font-mono)')
                    .text(p.label);

                if (!Number.isFinite(p.value)) {
                    svg.append('text')
                        .attr('x', pad.left + 4).attr('y', y + rowH / 2 + 4)
                        .attr('fill', axis).style('font-size', '10px').text('tidak ada nilai');
                    return;
                }

                svg.append('rect')
                    .attr('x', pad.left + Math.min(x(0), x(p.value)))
                    .attr('y', y + 5)
                    .attr('width', Math.max(2, Math.abs(x(p.value) - x(0))))
                    .attr('height', rowH - 12)
                    .attr('rx', 4)
                    .attr('fill', pen)
                    // The best run keeps full strength; the rest recede. One encoding, no legend.
                    .attr('fill-opacity', isBest ? 1 : 0.45)
                    .on('pointerenter', () => tip.show(
                        `<b>${p.value.toFixed(4)}</b><br>${p.label}${isBest ? '<br>terbaik sejauh ini' : ''}`,
                        pad.left + Math.abs(x(p.value) - x(0)) + 12, y + rowH / 2))
                    .on('pointerleave', () => tip.hide());

                svg.append('text')
                    .attr('x', width - pad.right + 6).attr('y', y + rowH / 2 + 4)
                    .attr('fill', isBest ? ink : axis)
                    .style('font-size', '10px').style('font-weight', isBest ? '600' : '400')
                    .style('font-family', 'var(--font-mono)')
                    .text(p.value.toFixed(4));
            });

            svg.append('line')
                .attr('x1', pad.left + x(0)).attr('x2', pad.left + x(0))
                .attr('y1', pad.top).attr('y2', height - 8)
                .attr('stroke', axis).attr('stroke-width', 1);
        },

        /** Redraws every chart in a container — used on theme change and resize. */
        redrawAll(container) {
            if (container && container.__redraw) {
                container.__redraw();
            }
        }
    };
})();
