import Vue from 'vue';
import Router from 'vue-router';
import PatientClassifier from '@/components/PatientClassifier';
import VesselsNeighborhood from '../components/VesselsNeighborhood.vue';

Vue.use(Router);

export default new Router({
  routes: [
    {
      path: '/',
      name: 'PatientClassifier',
      component: PatientClassifier,
    },
    {
      path: '/vessels-neighborhood',
      name: 'VesselsNeighborhood',
      component: VesselsNeighborhood,
    },
  ],
});
