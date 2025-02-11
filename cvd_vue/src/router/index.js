import Vue from 'vue';
import Router from 'vue-router';
import VesselsNeighborhood from '../components/VesselsNeighborhood';
import VesselsRules from '../components/VesselsRules';
import LandingPage from '../components/LandingPage';
import COVIDNeighborhood from '../components/COVIDNeighborhood';
import COVIDRules from '../components/COVIDRules';

Vue.use(Router);

export default new Router({
  routes: [
    {
      path: '/',
      name: 'LandingPage',
      component: LandingPage,
    },
    {
      path: '/vessels-neighborhood',
      name: 'VesselsNeighborhood',
      component: VesselsNeighborhood,
    },
    {
      path: '/vessels-rules',
      name: 'VesselsRules',
      component: VesselsRules,
    },
    {
      path: '/covid-neighborhood',
      name: 'COVIDNeighborhood',
      component: COVIDNeighborhood,
    },
    {
      path: '/covid-rules',
      name: 'COVIDRules',
      component: COVIDRules,
    },
  ],
});
