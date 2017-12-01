import React, { Component } from 'react';
import _ from 'lodash';
const d3 = require('d3');
const d3Tip = require('d3-tip');

class App extends Component {
  constructor() {
    super();
    this.state = {};
  }

  componentDidMount() {
    fetch('http://localhost:5000').then(res => res.json()).then(data => {
      this.setState({data}, this.mountChart);
    });
  }

  render() {
    const width = window.innerWidth - 50,
      height = window.innerHeight - 50;
    return (
      <div>
        <svg width={width} height={height} />
      </div>
    );
  }

  mountChart = () => {
    let {data} = this.state;
    let svg = d3.select("svg"),
      margin = {top: 20, right: 20, bottom: 30, left: 50},
      width = +svg.attr("width") - margin.left - margin.right,
      height = +svg.attr("height") - margin.top - margin.bottom,
      g = svg.append("g").attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    data = _.filter(data, d => _.uniq(d.actions).length > 1);
    data = _.sortBy(data, 'reward_avg').reverse().slice(0,2);
    data.forEach(d => {
      d.rewards = d.rewards.map((v,i) => ({y:v, x:i, parent:d}));
    });
    let all_rewards = _(data).map('rewards').flatten().value();

    let colorScale = d3.scaleOrdinal(d3.schemeCategory10)
      .domain([0, data.length]);

    let x = d3.scaleLinear()
      .rangeRound([0, width])
      .domain([0, data[0].rewards.length]);
    let y = d3.scaleLinear()
      .rangeRound([height, 0])
      .domain(d3.extent(all_rewards, d => d.y));

    let axes = {
      x: d3.axisBottom(x),
      y: d3.axisLeft(y)
    };
    let axes_g = {
      x: g.append("g")
        .attr("transform", "translate(0," + height + ")")
        .call(axes.x),
      y: g.append("g")
        .call(axes.y)
    };
    axes_g.y.append("text")
      .attr("fill", "#000")
      .attr("transform", "rotate(-90)")
      .attr("y", 6)
      .attr("dy", "0.71em")
      .attr("text-anchor", "end")
      .text("Reward");

    let line = d3.line()
        .x(d => x(d.x))
        .y(d => y(d.y));

    data.forEach((row, i) => {
      g.datum(row.rewards)
        .append("path")
        .classed('line', true)
        .attr("fill", "none")
        .attr("stroke", () => colorScale(i))
        .attr("stroke-linejoin", "round")
        .attr("stroke-linecap", "round")
        .attr("stroke-width", 1.5)
        .attr("d", line);
    });

    // Add a 0-line
    let zeroLine = [
      {x:0,y:0},
      {x:data[0].rewards.length, y:0}
    ];
    g.datum(zeroLine)
      .append('path')
      .classed('line', true)
      .attr("fill", "none")
      .attr("stroke", 'black')
      .attr("stroke-linejoin", "round")
      .attr("stroke-linecap", "round")
      .attr("stroke-width", 1.5)
      .attr("d", line);

    let tip = d3Tip()
      .attr('class', 'd3-tip')
      .direction('sw')
      .html(d => {
        let str = `
          reward: ${d.y}<br/>
          reward_avg: ${d.parent.reward_avg}<br/>
          source: ${d.parent.source}<br/>
          uniques: ${_.uniq(d.parent.actions).length}<br/><br/>
        `;

        return str + JSON.stringify(d.parent.hypers, null, 2).replace(/\n/g,'<br/>')
      });

    g.call(tip);
    g.selectAll("dot").data(all_rewards)
      .enter()
        .append("circle")
        .classed('dot', true)
        .attr("r", 2)
        .attr("cx", d => x(d.x))
        .attr("cy", d => y(d.y))
        .on("mouseover", tip.show)
        .on("mouseout", tip.hide);

    let zoom = d3.zoom()
      .scaleExtent([0,500])
      .on("zoom", zooming);
    svg.call(zoom);
    function zooming() {
      g.selectAll('.dot').attr("transform", d3.event.transform);
      g.selectAll('.line').attr("transform", d3.event.transform);
      axes_g.x.call(axes.x.scale(d3.event.transform.rescaleX(x)));
      axes_g.y.call(axes.y.scale(d3.event.transform.rescaleY(y)));
    }
  };
}

export default App;
