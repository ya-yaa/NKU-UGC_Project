function initResultChart(containerId, metrics) {
  const container = document.getElementById(containerId);
  if (!container || typeof echarts === 'undefined' || !metrics) {
    return;
  }

  const chart = echarts.init(container);
  chart.setOption({
    backgroundColor: 'transparent',
    tooltip: {},
    radar: {
      indicator: [
        { name: '压缩率', max: 100 },
        { name: '准确率', max: 1 },
        { name: 'Alpha', max: 1 },
        { name: 'Bin Width', max: 1 }
      ],
      splitArea: { areaStyle: { color: ['rgba(181,74,45,0.05)', 'rgba(39,76,94,0.05)'] } }
    },
    series: [
      {
        type: 'radar',
        data: [
          {
            value: [
              metrics.reduction || 0,
              metrics.accuracy || 0,
              metrics.alpha || 0,
              metrics.bin_width || 0
            ],
            areaStyle: { color: 'rgba(181,74,45,0.25)' },
            lineStyle: { color: '#b54a2d' },
            symbol: 'circle',
            itemStyle: { color: '#274c5e' }
          }
        ]
      }
    ]
  });
}
