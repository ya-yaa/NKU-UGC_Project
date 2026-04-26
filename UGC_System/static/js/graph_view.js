function initGraphPreview(containerId, detailId, originalContainerId, originalDetailId, graphPreview, mappingApi) {
  const container = document.getElementById(containerId);
  const detail = document.getElementById(detailId);
  const originalContainer = document.getElementById(originalContainerId);
  const originalDetail = document.getElementById(originalDetailId);
  const coarseLoading = document.getElementById('graph-preview-loading');
  const originalLoading = document.getElementById('original-graph-preview-loading');
  const coarseFullscreenButton = document.getElementById('graph-preview-fullscreen');
  const originalFullscreenButton = document.getElementById('original-graph-preview-fullscreen');
  const edgePercentInput = document.getElementById('edge-percentile-input');
  const edgePercentApply = document.getElementById('edge-percentile-apply');
  const graphVisibilityNote = document.getElementById('graph-visibility-note');
  if (!container || !detail || typeof cytoscape === 'undefined' || !graphPreview) {
    return;
  }

  const nodeUrlTemplate = mappingApi && mappingApi.nodeUrlTemplate;
  const edgeUrlTemplate = mappingApi && mappingApi.edgeUrlTemplate;
  const allEdges = (graphPreview.all_edges || []).map((edge) => ({ data: edge.data || edge }));
  const fallbackEdges = (graphPreview.edges || []).map((edge) => ({ data: edge.data || edge }));
  const totalEdgeCount = allEdges.length || fallbackEdges.length;
  const nodeCache = new Map();
  const edgeCache = new Map();
  const coarseNodeCount = (graphPreview.nodes || []).length;

  const edgeMetaMap = new Map();
  [...allEdges, ...fallbackEdges].forEach((edge) => {
    if (edge && edge.data && edge.data.id && !edgeMetaMap.has(edge.data.id)) {
      edgeMetaMap.set(edge.data.id, edge.data);
    }
  });

  let originalCy = null;
  const coarseShell = container ? container.closest('.graph-loading-shell') : null;
  const originalShell = originalContainer ? originalContainer.closest('.graph-loading-shell') : null;
  const lightCoarseLayoutOptions = {
    name: 'concentric',
    animate: false,
    padding: 24,
    minNodeSpacing: 8,
    concentric(node) {
      return Number(node.data('size') || node.data('count') || 1);
    },
    levelWidth() {
      return 10;
    }
  };
  const forceDirectedLayoutOptions = {
    name: 'cose',
    animate: false,
    padding: 24,
    idealEdgeLength: 60
  };
  const coarseLayoutOptions = coarseNodeCount <= 1500 ? forceDirectedLayoutOptions : lightCoarseLayoutOptions;
  const setCoarseLoading = (visible) => {
    if (!coarseLoading) {
      return;
    }
    coarseLoading.classList.toggle('is-hidden', !visible);
  };
  const setOriginalGraphLoading = (visible) => {
    if (!originalLoading) {
      return;
    }
    originalLoading.classList.toggle('is-hidden', !visible);
  };
  if (originalContainer) {
    originalCy = cytoscape({
      container: originalContainer,
      elements: [],
      style: [
        {
          selector: 'node',
          style: {
            label: 'data(label)',
            'font-size': 8,
            width: 12,
            height: 12,
            'background-color': '#7e9daa',
            color: '#1e2a2f',
            'border-width': 0,
            'border-color': '#ffffff'
          }
        },
        {
          selector: 'node[kind = "member"]',
          style: {
            'background-color': '#d4542f',
            width: 18,
            height: 18,
            color: '#ffffff'
          }
        },
        {
          selector: 'node[group = "source"]',
          style: {
            'background-color': '#d4542f'
          }
        },
        {
          selector: 'node[group = "target"]',
          style: {
            'background-color': '#1f6f8b'
          }
        },
        {
          selector: 'node.is-focused',
          style: {
            'border-width': 3,
            'border-color': '#1d3557'
          }
        },
        {
          selector: 'edge',
          style: {
            width: 1,
            'line-color': '#b8c6cd',
            opacity: 0.8
          }
        },
        {
          selector: 'edge.is-focused',
          style: {
            width: 4,
            'line-color': '#1d3557',
            'target-arrow-color': '#1d3557',
            opacity: 1
          }
        }
      ],
      layout: { name: 'grid', fit: true, padding: 20 }
    });
  }

  const renderDetail = (nodeId) => {
    const meta = (graphPreview.details || {})[nodeId] || {};
    const neighbors = meta.neighbors || [];
    const topLabels = meta.top_labels || [];
    const memberPreview = meta.member_preview || [];
    const purity = Number(meta.purity);
    const purityText = Number.isFinite(purity) ? purity.toFixed(4) : '-';
    detail.innerHTML = `
      <strong>${meta.title || nodeId}</strong>
      <p>聚合规模：${meta.count || meta.member_count || '-'}</p>
      <p>粗图标签：${meta.coarse_label ?? '-'}</p>
      <p>Supernode Purity：${purityText}</p>
      <p>主导类别：${meta.dominant_label ?? '-'}</p>
      <p>成员节点预览：${memberPreview.length ? memberPreview.join(', ') : '暂无'}</p>
      <p>主标签分布：${topLabels.length ? topLabels.map((item) => `L${item.label}:${item.count}`).join(' , ') : '暂无'}</p>
      <p>邻接 Supernode：${neighbors.length ? neighbors.slice(0, 12).map((item) => `${item.id} (${item.weight.toFixed ? item.weight.toFixed(1) : item.weight})`).join(' , ') : '暂无'}</p>
    `;
  };

  const renderEdgeDetail = (edgeId) => {
    const edgeMeta = edgeMetaMap.get(edgeId) || {};
    detail.innerHTML = `
      <strong>Coarse Edge ${edgeId}</strong>
      <p>起点 supernode：${edgeMeta.source || '-'}</p>
      <p>终点 supernode：${edgeMeta.target || '-'}</p>
      <p>边权：${edgeMeta.weight ?? '-'}</p>
      <p>点击下方原图中的边，可反向高亮这条 coarse edge。</p>
    `;
  };

  const cy = cytoscape({
    container,
    elements: [...(graphPreview.nodes || []), ...fallbackEdges],
    style: [
      {
        selector: 'node',
        style: {
          'background-color': '#b54a2d',
          label: 'data(label)',
          color: '#ffffff',
          'font-size': 10,
          'text-valign': 'center',
          'text-halign': 'center',
          width: 'mapData(size, 1, 300, 22, 68)',
          height: 'mapData(size, 1, 300, 22, 68)',
          'border-width': 0,
          'border-color': '#ffe6d5'
        }
      },
      {
        selector: 'node.is-focused',
        style: {
          'border-width': 4,
          'border-color': '#1d3557',
          'overlay-color': '#1d3557',
          'overlay-opacity': 0.08
        }
      },
      {
        selector: 'edge',
        style: {
          width: 'mapData(weight, 1, 1000, 1, 6)',
          'line-color': '#7e9daa',
          'curve-style': 'bezier',
          opacity: 0.7
        }
      },
      {
        selector: 'edge.is-focused',
        style: {
          width: 8,
          'line-color': '#1d3557',
          opacity: 1,
          'z-index': 20
        }
      }
    ],
    layout: coarseLayoutOptions
  });

  const updateGraphVisibilityNote = (percentile, visibleCount) => {
    if (!graphVisibilityNote) {
      return;
    }
    const totalCount = totalEdgeCount || visibleCount || 0;
    graphVisibilityNote.textContent = `当前按边权排名显示前 ${percentile}% 的 coarse edges，共 ${visibleCount} / ${totalCount} 条。`;
  };

  const getVisibleEdges = (percentileValue) => {
    const parsedValue = Number(percentileValue);
    const percentile = Number.isFinite(parsedValue) ? Math.min(100, Math.max(1, parsedValue)) : 5;
    const edgePool = allEdges.length ? allEdges : fallbackEdges;
    if (!edgePool.length) {
      return { percentile, edges: [] };
    }
    const visibleCount = Math.max(1, Math.ceil(edgePool.length * percentile / 100));
    return { percentile, edges: edgePool.slice(0, visibleCount) };
  };

  const applyEdgePercentile = () => {
    const visible = getVisibleEdges(edgePercentInput ? edgePercentInput.value : 5);
    if (edgePercentInput) {
      edgePercentInput.value = String(visible.percentile);
    }
    setCoarseLoading(true);
    cy.elements().remove();
    cy.add([...(graphPreview.nodes || []), ...visible.edges]);
    const layout = cy.layout(coarseLayoutOptions);
    layout.one('layoutstop', () => {
      setCoarseLoading(false);
    });
    layout.run();
    updateGraphVisibilityNote(visible.percentile, visible.edges.length);
    return visible;
  };

  const focusCoarseSupernode = (nodeId) => {
    cy.nodes().removeClass('is-focused');
    cy.edges().removeClass('is-focused');
    const node = cy.getElementById(nodeId);
    if (!node || node.empty()) {
      return;
    }
    node.addClass('is-focused');
    cy.animate({
      center: { eles: node },
      zoom: Math.max(cy.zoom(), 1.1)
    }, {
      duration: 250
    });
    renderDetail(nodeId);
  };

  const focusCoarseEdge = (edgeId) => {
    cy.nodes().removeClass('is-focused');
    cy.edges().removeClass('is-focused');
    const edge = cy.getElementById(edgeId);
    if (!edge || edge.empty()) {
      return;
    }
    edge.addClass('is-focused');
    cy.animate({
      center: { eles: edge.connectedNodes().union(edge) },
      zoom: Math.max(cy.zoom(), 1.05)
    }, {
      duration: 250
    });
    renderEdgeDetail(edgeId);
  };

  const renderOriginalGraph = (subgraph, fallbackHtml) => {
    if (!originalCy || !originalDetail) {
      return;
    }
    if (!subgraph) {
      originalCy.elements().remove();
      setOriginalGraphLoading(false);
      originalDetail.innerHTML = fallbackHtml;
      return;
    }
    setOriginalGraphLoading(true);
    originalCy.elements().remove();
    originalCy.add([...(subgraph.nodes || []), ...(subgraph.edges || [])]);
    const layout = originalCy.layout({ name: 'cose', animate: false, padding: 20, idealEdgeLength: 28 });
    layout.one('layoutstop', () => {
      setOriginalGraphLoading(false);
    });
    layout.run();
  };

  const setOriginalLoading = (message) => {
    setOriginalGraphLoading(true);
    if (originalCy) {
      originalCy.elements().remove();
    }
    if (originalDetail) {
      originalDetail.innerHTML = `<strong>${message}</strong><p>正在从后端按需加载原图局部映射。</p>`;
    }
  };

  const fetchNodeSubgraph = async (nodeId) => {
    if (nodeCache.has(nodeId)) {
      return nodeCache.get(nodeId);
    }
    if (!nodeUrlTemplate) {
      return null;
    }
    const response = await fetch(nodeUrlTemplate.replace('__SUPERNODE_ID__', encodeURIComponent(nodeId)), {
      headers: { Accept: 'application/json' },
      cache: 'no-store'
    });
    if (!response.ok) {
      throw new Error('node mapping fetch failed');
    }
    const payload = await response.json();
    nodeCache.set(nodeId, payload);
    return payload;
  };

  const fetchEdgeSubgraph = async (edgeId) => {
    if (edgeCache.has(edgeId)) {
      return edgeCache.get(edgeId);
    }
    if (!edgeUrlTemplate) {
      return null;
    }
    const response = await fetch(edgeUrlTemplate.replace('__EDGE_ID__', encodeURIComponent(edgeId)), {
      headers: { Accept: 'application/json' },
      cache: 'no-store'
    });
    if (!response.ok) {
      throw new Error('edge mapping fetch failed');
    }
    const payload = await response.json();
    edgeCache.set(edgeId, payload);
    return payload;
  };

  const bindOriginalNodeEvents = (defaultDetailHtml) => {
    if (!originalCy || !originalDetail) {
      return;
    }
    originalCy.off('tap', 'node');
    originalCy.off('tap', 'edge');
    originalCy.on('tap', 'node', (event) => {
      originalCy.nodes().removeClass('is-focused');
      originalCy.edges().removeClass('is-focused');
      const originalNode = event.target;
      originalNode.addClass('is-focused');
      const ownerSupernode = originalNode.data('supernode_id');
      if (ownerSupernode) {
        focusCoarseSupernode(ownerSupernode);
      }
      const classLabel = originalNode.data('class_label');
      originalDetail.innerHTML = `
        <strong>原始节点 ${originalNode.data('label')}</strong>
        <p>类别标签：${classLabel ?? '-'}</p>
        <p>所属 supernode：${ownerSupernode || '未知'}</p>
        <p>节点类型：${originalNode.data('kind') === 'member' ? '当前 supernode 成员' : (originalNode.data('group') === 'source' ? '边起点侧' : originalNode.data('group') === 'target' ? '边终点侧' : '上下文邻居')}</p>
      `;
    });
    originalCy.on('tap', 'edge', (event) => {
      originalCy.nodes().removeClass('is-focused');
      originalCy.edges().removeClass('is-focused');
      const originalEdge = event.target;
      originalEdge.addClass('is-focused');
      const ownerEdge = originalEdge.data('coarse_edge_id');
      if (ownerEdge) {
        focusCoarseEdge(ownerEdge);
      }
      originalDetail.innerHTML = `
        <strong>原始边 ${originalEdge.data('source')} - ${originalEdge.data('target')}</strong>
        <p>映射 coarse edge：${ownerEdge || '当前局部子图内无直接 coarse edge 映射'}</p>
      `;
    });
    if (defaultDetailHtml) {
      originalDetail.innerHTML = defaultDetailHtml;
    }
  };

  const renderOriginalMapping = async (nodeId) => {
    setOriginalLoading(`正在加载 ${nodeId} 的原图映射`);
    try {
      const subgraph = await fetchNodeSubgraph(nodeId);
      if (!subgraph) {
        renderOriginalGraph(null, '<strong>暂无原图映射</strong><p>当前 supernode 没有可展示的原图局部子图。</p>');
        return;
      }
      renderOriginalGraph(subgraph, '');
      const meta = subgraph.meta || {};
      bindOriginalNodeEvents(`
        <strong>${nodeId} 的原图映射</strong>
        <p>高亮成员节点：${meta.member_count || 0}</p>
        <p>上下文邻居：${meta.context_count || 0}</p>
        <p>显示边数：${meta.displayed_edges || 0}</p>
        <p>点击下方原始节点，可反向高亮其所属 supernode。</p>
      `);
    } catch (error) {
      renderOriginalGraph(null, '<strong>原图映射加载失败</strong><p>当前 supernode 的局部映射没有成功返回，请稍后重试。</p>');
    }
  };

  const renderOriginalEdgeMapping = async (edgeId) => {
    setOriginalLoading(`正在加载 ${edgeId} 的边映射`);
    try {
      const subgraph = await fetchEdgeSubgraph(edgeId);
      if (!subgraph) {
        renderOriginalGraph(null, '<strong>暂无边映射</strong><p>当前 coarse edge 没有可展示的原始跨簇边。</p>');
        return;
      }
      renderOriginalGraph(subgraph, '');
      const meta = subgraph.meta || {};
      bindOriginalNodeEvents(`
        <strong>${meta.source_supernode || '-'} ↔ ${meta.target_supernode || '-'}</strong>
        <p>源 supernode 成员数：${meta.source_count || 0}</p>
        <p>目标 supernode 成员数：${meta.target_count || 0}</p>
        <p>显示原始跨簇边数：${meta.displayed_edges || 0}</p>
        <p>coarse edge 权重：${meta.coarse_weight ?? '-'}</p>
        <p>点击下方原始边，可反向高亮上方 coarse edge。</p>
      `);
    } catch (error) {
      renderOriginalGraph(null, '<strong>边映射加载失败</strong><p>当前 coarse edge 的局部映射没有成功返回，请稍后重试。</p>');
    }
  };

  cy.on('tap', 'node', (event) => {
    const nodeId = event.target.id();
    focusCoarseSupernode(nodeId);
    renderOriginalMapping(nodeId);
  });
  cy.on('tap', 'edge', (event) => {
    const edgeId = event.target.id();
    focusCoarseEdge(edgeId);
    renderOriginalEdgeMapping(edgeId);
  });

  if (edgePercentApply) {
    edgePercentApply.addEventListener('click', () => {
      applyEdgePercentile();
    });
  }
  if (edgePercentInput) {
    edgePercentInput.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') {
        applyEdgePercentile();
      }
    });
  }

  applyEdgePercentile();

  const requestFullscreenFor = (element, onResize) => {
    if (!element || !document.fullscreenEnabled) {
      return;
    }
    if (document.fullscreenElement === element) {
      document.exitFullscreen().catch(() => {});
      return;
    }
    element.requestFullscreen().then(() => {
      if (onResize) {
        window.setTimeout(onResize, 120);
      }
    }).catch(() => {});
  };

  if (coarseFullscreenButton) {
    coarseFullscreenButton.addEventListener('click', () => {
      requestFullscreenFor(coarseShell, () => {
        cy.resize();
        cy.fit(undefined, 24);
      });
    });
  }

  if (originalFullscreenButton) {
    originalFullscreenButton.addEventListener('click', () => {
      requestFullscreenFor(originalShell, () => {
        if (originalCy) {
          originalCy.resize();
          originalCy.fit(undefined, 24);
        }
      });
    });
  }

  document.addEventListener('fullscreenchange', () => {
    cy.resize();
    if (originalCy) {
      originalCy.resize();
    }
  });

  const firstNode = cy.nodes()[0];
  if (firstNode) {
    focusCoarseSupernode(firstNode.id());
    renderOriginalMapping(firstNode.id());
  }
}
