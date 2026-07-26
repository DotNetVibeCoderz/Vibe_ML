/* ============================================================================
   Experiment designer canvas.

   Edges are SVG (D3 draws and animates the paths); nodes are HTML positioned
   over the top. Splitting it this way means node chrome reuses the same CSS as
   the rest of the app, while the edges get D3's path interpolation for the
   run-trace animation — an SVG-only canvas would mean rebuilding every card in
   <foreignObject>, and an HTML-only one could not draw the traces.
   ========================================================================= */

(function () {
    const PEN = {
        DataInput: 'blue', DataTransform: 'teal', LlmAction: 'violet',
        Algorithm: 'amber', Training: 'rose', Evaluation: 'lime',
        Script: 'ink', Output: 'blue'
    };

    const NODE_W = 188;
    const HEAD_H = 34;

    class Canvas {
        constructor(root, dotnet) {
            this.root = root;
            this.dotnet = dotnet;
            this.graph = { nodes: [], edges: [], view: { panX: 0, panY: 0, zoom: 1 } };
            this.states = {};
            this.selected = null;
            this.linking = null;

            this.viewport = root.querySelector('.dz-viewport');
            this.world = root.querySelector('.dz-world');
            // The group inside the <svg>, not the <svg> itself — see the note in Designer.razor.
            this.edges = d3.select(root).select('.dz-edge-layer');
            this.nodeLayer = root.querySelector('.dz-nodes');

            this._setupZoom();
            this._setupDrop();
            this._setupLinking();
        }

        // ---------------------------------------------------------- zoom/pan

        _setupZoom() {
            this.zoom = d3.zoom()
                .scaleExtent([0.25, 2.5])
                // Dragging a node or pulling a link must not also pan the canvas.
                .filter(e => !e.target.closest('.node') && !e.target.closest('.port') && !e.button)
                .on('zoom', e => {
                    const t = e.transform;
                    // Both layers take the same transform, expressed the way each understands it.
                    this.world.style.transform = `translate(${t.x}px, ${t.y}px) scale(${t.k})`;
                    this.edges.attr('transform', `translate(${t.x}, ${t.y}) scale(${t.k})`);
                    this.transform = t;
                });

            d3.select(this.viewport).call(this.zoom).on('dblclick.zoom', null);
            this.transform = d3.zoomIdentity;
        }

        setView(view) {
            const t = d3.zoomIdentity.translate(view.panX || 0, view.panY || 0).scale(view.zoom || 1);
            d3.select(this.viewport).call(this.zoom.transform, t);
        }

        zoomBy(factor) {
            d3.select(this.viewport).transition().duration(180).call(this.zoom.scaleBy, factor);
        }

        fit() {
            if (!this.graph.nodes.length) return;

            const xs = this.graph.nodes.map(n => n.x);
            const ys = this.graph.nodes.map(n => n.y);
            const minX = Math.min(...xs) - 60, maxX = Math.max(...xs) + NODE_W + 60;
            const minY = Math.min(...ys) - 60, maxY = Math.max(...ys) + 140;

            const w = this.viewport.clientWidth, h = this.viewport.clientHeight;
            const k = Math.min(2, Math.max(0.25, Math.min(w / (maxX - minX), h / (maxY - minY))));

            const t = d3.zoomIdentity
                .translate(w / 2 - k * (minX + maxX) / 2, h / 2 - k * (minY + maxY) / 2)
                .scale(k);

            d3.select(this.viewport).transition().duration(260).call(this.zoom.transform, t);
        }

        // ------------------------------------------------- drop from palette

        _setupDrop() {
            this.viewport.addEventListener('dragover', e => {
                e.preventDefault();
                e.dataTransfer.dropEffect = 'copy';
                this.viewport.classList.add('is-dropping');
            });

            this.viewport.addEventListener('dragleave', () => this.viewport.classList.remove('is-dropping'));

            this.viewport.addEventListener('drop', e => {
                e.preventDefault();
                this.viewport.classList.remove('is-dropping');

                const moduleId = e.dataTransfer.getData('text/blazorml-module');
                if (!moduleId) return;

                const rect = this.viewport.getBoundingClientRect();
                const p = this.transform.invert([e.clientX - rect.left, e.clientY - rect.top]);

                this.dotnet.invokeMethodAsync('OnModuleDropped', moduleId,
                    Math.round(p[0] - NODE_W / 2), Math.round(p[1] - HEAD_H / 2));
            });
        }

        // ------------------------------------------------------ port linking

        _setupLinking() {
            this.root.addEventListener('pointerdown', e => {
                const port = e.target.closest('.port--out');
                if (!port) return;

                e.preventDefault();
                e.stopPropagation();

                this.linking = {
                    nodeId: port.closest('.node').dataset.id,
                    port: Number(port.dataset.port)
                };

                this.root.classList.add('is-linking');
                this._drawPendingLink(e);
            });

            this.root.addEventListener('pointermove', e => {
                if (this.linking) this._drawPendingLink(e);
            });

            this.root.addEventListener('pointerup', e => {
                if (!this.linking) return;

                const target = e.target.closest('.port--in');
                if (target) {
                    this.dotnet.invokeMethodAsync('OnEdgeCreated',
                        this.linking.nodeId, this.linking.port,
                        target.closest('.node').dataset.id, Number(target.dataset.port));
                }

                this.linking = null;
                this.root.classList.remove('is-linking');
                this.edges.select('.dz-pending').remove();
            });
        }

        _drawPendingLink(event) {
            const from = this._portCentre(this.linking.nodeId, this.linking.port, 'out');
            if (!from) return;

            const rect = this.viewport.getBoundingClientRect();
            const to = this.transform.invert([event.clientX - rect.left, event.clientY - rect.top]);

            let path = this.edges.select('.dz-pending');
            if (path.empty()) {
                path = this.edges.append('path').attr('class', 'dz-pending');
            }

            path.attr('d', this._route(from, { x: to[0], y: to[1] }));
        }

        // ------------------------------------------------------------ render

        render(graph, states) {
            this.graph = graph;
            this.states = states || {};

            this._renderNodes();
            this._renderEdges();
        }

        _renderNodes() {
            const seen = new Set();

            for (const node of this.graph.nodes) {
                seen.add(node.id);

                let el = this.nodeLayer.querySelector(`[data-id="${node.id}"]`);
                if (!el) {
                    el = this._createNode(node);
                    this.nodeLayer.appendChild(el);
                }

                el.style.left = node.x + 'px';
                el.style.top = node.y + 'px';
                el.style.setProperty('--pen', `var(--pen-${PEN[node.category] || 'ink'})`);
                el.classList.toggle('is-selected', node.id === this.selected);

                const state = this.states[node.id];
                el.dataset.state = state ? state.toLowerCase() : '';
                el.querySelector('.node-title').textContent = node.label;

                const badge = el.querySelector('.node-badge');
                badge.textContent = node.badge || '';
                badge.hidden = !node.badge;
            }

            // Remove nodes that are no longer in the graph.
            for (const el of [...this.nodeLayer.children]) {
                if (!seen.has(el.dataset.id)) el.remove();
            }
        }

        _createNode(node) {
            const el = document.createElement('div');
            el.className = 'node';
            el.dataset.id = node.id;

            el.innerHTML = `
                <div class="node-ports node-ports--in">
                    ${node.inputs.map((p, i) =>
                        `<span class="port port--in" data-port="${i}" title="${p}"></span>`).join('')}
                </div>
                <div class="node-head">
                    <span class="node-title"></span>
                    <span class="node-state" aria-hidden="true"></span>
                </div>
                <div class="node-badge mono" hidden></div>
                <div class="node-ports node-ports--out">
                    ${node.outputs.map((p, i) =>
                        `<span class="port port--out" data-port="${i}" title="${p}"></span>`).join('')}
                </div>`;

            el.addEventListener('pointerdown', e => {
                if (e.target.closest('.port')) return;
                this.select(node.id);
            });

            d3.select(el).call(d3.drag()
                .filter(e => !e.target.closest('.port'))
                .on('start', function () { this.classList.add('is-dragging'); })
                .on('drag', e => {
                    const current = this.graph.nodes.find(n => n.id === node.id);
                    if (!current) return;

                    current.x += e.dx / this.transform.k;
                    current.y += e.dy / this.transform.k;

                    el.style.left = current.x + 'px';
                    el.style.top = current.y + 'px';
                    this._renderEdges();
                })
                .on('end', (e, d) => {
                    el.classList.remove('is-dragging');
                    const current = this.graph.nodes.find(n => n.id === node.id);
                    this.dotnet.invokeMethodAsync('OnNodeMoved', node.id,
                        Math.round(current.x), Math.round(current.y));
                }));

            return el;
        }

        select(id) {
            this.selected = id;
            for (const el of this.nodeLayer.children) {
                el.classList.toggle('is-selected', el.dataset.id === id);
            }
            this.dotnet.invokeMethodAsync('OnNodeSelected', id);
        }

        _portCentre(nodeId, port, side) {
            const node = this.graph.nodes.find(n => n.id === nodeId);
            if (!node) return null;

            const count = side === 'in' ? node.inputs.length : node.outputs.length;
            if (!count) return null;

            // Ports are spread evenly across the card's width so an edge lands on the port it
            // belongs to rather than on the middle of the card.
            const step = NODE_W / (count + 1);

            return {
                x: node.x + step * (port + 1),
                y: side === 'in' ? node.y : node.y + HEAD_H + 18
            };
        }

        /* Orthogonal route with rounded corners: the plotted look, and easier to follow than a
           bezier when several edges converge on one node. */
        _route(from, to) {
            const dy = to.y - from.y;
            const midY = from.y + dy / 2;
            const r = Math.min(14, Math.abs(dy) / 2, Math.abs(to.x - from.x) / 2 || 14);

            if (Math.abs(to.x - from.x) < 2) {
                return `M${from.x},${from.y} L${to.x},${to.y}`;
            }

            const sweepDown = dy > 0 ? 1 : -1;
            const sweepRight = to.x > from.x ? 1 : -1;

            return [
                `M${from.x},${from.y}`,
                `L${from.x},${midY - r * sweepDown}`,
                `Q${from.x},${midY} ${from.x + r * sweepRight},${midY}`,
                `L${to.x - r * sweepRight},${midY}`,
                `Q${to.x},${midY} ${to.x},${midY + r * sweepDown}`,
                `L${to.x},${to.y}`
            ].join(' ');
        }

        _renderEdges() {
            const data = this.graph.edges.map(edge => {
                const from = this._portCentre(edge.sourceNodeId, edge.sourcePort, 'out');
                const to = this._portCentre(edge.targetNodeId, edge.targetPort, 'in');
                if (!from || !to) return null;

                const source = this.graph.nodes.find(n => n.id === edge.sourceNodeId);

                return {
                    id: edge.id,
                    d: this._route(from, to),
                    pen: PEN[source?.category] || 'ink',
                    active: this.states[edge.sourceNodeId] === 'Running'
                        || this.states[edge.targetNodeId] === 'Running'
                };
            }).filter(Boolean);

            const paths = this.edges.selectAll('path.dz-edge').data(data, d => d.id);

            paths.exit().remove();

            const entered = paths.enter().append('path')
                .attr('class', 'dz-edge')
                .attr('marker-end', 'url(#dz-arrow)');

            entered.merge(paths)
                .attr('d', d => d.d)
                .style('stroke', d => `var(--pen-${d.pen})`)
                .classed('is-active', d => d.active);

            // Click an edge to remove it. The hit area is a wide invisible path on top, because
            // a 2px line is not a realistic click target.
            const hits = this.edges.selectAll('path.dz-hit').data(data, d => d.id);
            hits.exit().remove();
            hits.enter().append('path')
                .attr('class', 'dz-hit')
                .on('click', (e, d) => this.dotnet.invokeMethodAsync('OnEdgeDeleted', d.id))
                .merge(hits)
                .attr('d', d => d.d);
        }

        setStates(states) {
            this.states = states || {};
            this._renderNodes();
            this._renderEdges();
        }

        destroy() {
            d3.select(this.viewport).on('.zoom', null);
        }
    }

    const instances = new Map();

    window.designer = {
        init(root, dotnet) {
            if (instances.has(root)) instances.get(root).destroy();

            const canvas = new Canvas(root, dotnet);
            instances.set(root, canvas);
            return true;
        },

        render(root, graph, states) {
            instances.get(root)?.render(graph, states);
        },

        setStates(root, states) {
            instances.get(root)?.setStates(states);
        },

        setView(root, view) {
            instances.get(root)?.setView(view);
        },

        zoomBy(root, factor) {
            instances.get(root)?.zoomBy(factor);
        },

        fit(root) {
            instances.get(root)?.fit();
        },

        select(root, id) {
            instances.get(root)?.select(id);
        },

        destroy(root) {
            instances.get(root)?.destroy();
            instances.delete(root);
        }
    };
})();
