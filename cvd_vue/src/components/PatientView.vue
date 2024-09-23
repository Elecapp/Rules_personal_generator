<template>
  <div class="mb-3">
    <h4>{{title}}</h4>
    <b-list-group>
      <b-list-group-item v-for="(v, i) in data" :key="i">
        {{attributes[i]}}:{{v}}
      </b-list-group-item>
    </b-list-group>
  </div>
</template>

<script>


export default {
  name: 'PatientView',
  data() {
    return {
      attributes: ['Age', 'Hearthrate', 'SBP', 'Creatinine', 'KILLIP', 'CAA', 'TN', 'ST'],
    };
  },
  props: {
    data: Array,
    title: {
      type: String,
      default: 'Patient',
    },
  },
};
</script>

<style scoped>

</style>
